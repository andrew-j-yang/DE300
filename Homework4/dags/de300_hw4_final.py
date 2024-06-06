from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import boto3
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
import logging

# Configuration for the DAG
default_arguments = {
    'owner': 'AndrewYang',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Initialize the DAG
dag = DAG(
    'andrew_hw4',
    default_args=default_arguments,
    description='andrew_hw4',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

# Load data from S3 function
def fetch_data_from_s3(**kwargs):
    s3_client = boto3.client('s3')
    bucket = 'andrewawsbucket'
    object_key = 'data/heart_disease.csv'
    local_path = '/tmp/heart_disease.csv'
    s3_client.download_file(bucket, object_key, local_path)
    logging.info(f"Downloaded {object_key} from {bucket} to {local_path}")
    return local_path

# Preprocessing and EDA function
def perform_eda(**kwargs):
    data = pd.read_csv("/tmp/heart_disease.csv")
    data = data.iloc[:899]

    columns_to_keep = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 
                       'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 
                       'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    df_cleaned = data[columns_to_keep]

    # Handle missing values
    df_cleaned['painloc'] = df_cleaned['painloc'].fillna(df_cleaned['painloc'].mode()[0])
    df_cleaned['painexer'] = df_cleaned['painexer'].fillna(df_cleaned['painexer'].mode()[0])
    df_cleaned.loc[df_cleaned['trestbps'] < 100, 'trestbps'] = 100
    df_cleaned.loc[df_cleaned['oldpeak'] < 0, 'oldpeak'] = 0
    df_cleaned.loc[df_cleaned['oldpeak'] > 4, 'oldpeak'] = 4
    df_cleaned['thaldur'] = df_cleaned['thaldur'].fillna(round(df_cleaned['thaldur'].mean(), 1))
    df_cleaned['thalach'] = df_cleaned['thalach'].fillna(round(df_cleaned['thalach'].mean(), 1))

    for column in ['fbs', 'prop', 'nitr', 'pro', 'diuretic', 'exang', 'slope']:
        mode_value = df_cleaned[column].mode()[0]
        df_cleaned[column] = df_cleaned[column].fillna(mode_value)
        df_cleaned.loc[df_cleaned[column] > 1, column] = mode_value

    continuous_cols = ['trestbps', 'oldpeak', 'thaldur', 'thalach']
    skew_values = df_cleaned[continuous_cols].skew()
    for col in continuous_cols:
        if abs(skew_values[col]) < 0.5:
            df_cleaned[col] = df_cleaned[col].fillna(round(df_cleaned[col].mean(), 1))
        else:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    # Impute smoke column using source 1
    def compute_smoking_percentage_1(age):
        if 15 <= age <= 17:
            return .016
        elif 18 <= age <= 24:
            return .073
        elif 25 <= age <= 34:
            return .109
        elif 35 <= age <= 44:
            return .109
        elif 45 <= age <= 54:
            return .138
        elif 55 <= age <= 64:
            return .149
        elif 65 <= age <= 74:
            return .087
        elif age >= 75:
            return .029
        else:
            return None
    df_cleaned['smoking_src1'] = df_cleaned.apply(lambda row: row['smoke'] if row['smoke'] in [0, 1] else compute_smoking_percentage_1(row['age']), axis=1)

    # Impute smoke column using source 2
    def compute_smoking_percentage_2(age, sex):
        if sex == 0:  # Female
            if 18 <= age <= 24:
                return .053
            elif 25 <= age <= 44:
                return .126
            elif 45 <= age <= 64:
                return .149
            elif age >= 65:
                return .083
        elif sex == 1:  # Male
            factor = .131 / .101
            if 18 <= age <= 24:
                return round(.053 * factor, 3)
            elif 25 <= age <= 44:
                return round(.126 * factor, 3)
            elif 45 <= age <= 64:
                return round(.149 * factor, 3)
            elif age >= 65:
                return round(.083 * factor, 3)
        return None
    df_cleaned['smoke_src2'] = df_cleaned.apply(lambda row: row['smoke'] if row['smoke'] in [0, 1] else compute_smoking_percentage_2(row['age'], row['sex']), axis=1)

    # Drop the original 'smoke' column
    df_cleaned.drop(columns=['smoke'], inplace=True)
    
    # Save the processed DataFrame
    df_cleaned.to_csv("/tmp/heart_disease_subset.csv", index=False)
    logging.info("Saved cleaned data to /tmp/heart_disease_subset.csv")
    return "/tmp/heart_disease_subset.csv"

# Create a Spark session
def initiate_spark_session():
    return SparkSession.builder.appName("Heart Disease Analysis").getOrCreate()

# Spark preprocessing function
def spark_eda(input_path, output_path, **kwargs):
    spark = initiate_spark_session()
    data = spark.read.csv(input_path, header=True, inferSchema=True)
    data = data.limit(899)

    columns_to_keep = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 
                       'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 
                       'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    df_cleaned = data.select(columns_to_keep)

    # Fill missing values
    for column in ['painloc', 'painexer', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'exang', 'slope']:
        mode_value = df_cleaned.groupBy(column).count().orderBy(col('count').desc()).first()[0]
        df_cleaned = df_cleaned.fillna({column: mode_value})

    df_cleaned = df_cleaned.withColumn('trestbps', when(col('trestbps') < 100, 100).otherwise(col('trestbps')))
    df_cleaned = df_cleaned.withColumn('oldpeak', when(col('oldpeak') < 0, 0).when(col('oldpeak') > 4, 4).otherwise(col('oldpeak')))
    mean_thaldur = df_cleaned.agg(mean('thaldur')).first()[0]
    mean_thalach = df_cleaned.agg(mean('thalach')).first()[0]
    df_cleaned = df_cleaned.fillna({'thaldur': mean_thaldur, 'thalach': mean_thalach})

    for column in ['fbs', 'prop', 'nitr', 'pro', 'diuretic']:
        mode_value = df_cleaned.groupBy(column).count().orderBy(col('count').desc()).first()[0]
        df_cleaned = df_cleaned.withColumn(column, when(col(column) > 1, mode_value).otherwise(col(column)))

    # Impute smoke column using source 1
    def compute_smoking_percentage_1(age):
        if 15 <= age <= 17:
            return .016
        elif 18 <= age <= 24:
            return .073
        elif 25 <= age <= 34:
            return .109
        elif 35 <= age <= 44:
            return .109
        elif 45 <= age <= 54:
            return .138
        elif 55 <= age <= 64:
            return .149
        elif 65 <= age <= 74:
            return .087
        elif age >= 75:
            return .029
        else:
            return None
    smoking_src1_udf = udf(compute_smoking_percentage_1)
    df_cleaned = df_cleaned.withColumn('smoking_src1', when(col('smoke').isin([0, 1]), col('smoke')).otherwise(smoking_src1_udf(col('age'))))

    # Impute smoke column using source 2
    def compute_smoking_percentage_2(age, sex):
        if sex == 0:
            if 18 <= age <= 24:
                return .053
            elif 25 <= age <= 44:
                return .126
            elif 45 <= age <= 64:
                return .149
            elif age >= 65:
                return .083
        elif sex == 1:
            factor = .131 / .101
            if 18 <= age <= 24:
                return round(.053 * factor, 3)
            elif 25 <= age <= 44:
                return round(.126 * factor, 3)
            elif 45 <= age <= 64:
                return round(.149 * factor, 3)
            elif age >= 65:
                return round(.083 * factor, 3)
        return None
    smoking_src2_udf = udf(compute_smoking_percentage_2)
    df_cleaned = df_cleaned.withColumn('smoke_src2', when(col('smoke').isin([0, 1]), col('smoke')).otherwise(smoking_src2_udf(col('age'), col('sex'))))

    df_cleaned = df_cleaned.drop('smoke')
    df_cleaned.write.csv(output_path, header=True, mode='overwrite')
    logging.info(f"Saved cleaned Spark data to {output_path}")
    return output_path

# Spark feature engineering function 1
def spark_feature_eng_1(input_path, output_path, **kwargs):
    spark = initiate_spark_session()
    data = spark.read.csv(input_path, header=True, inferSchema=True)
    data = data.withColumn('age_squared', col('age') ** 2)
    fe_path = output_path.replace('.csv', '_fe1.csv')
    data.write.csv(fe_path, header=True, mode='overwrite')
    logging.info(f"Saved feature engineered Spark data to {fe_path}")
    return fe_path

# Spark feature engineering function 2
def spark_feature_eng_2(input_path, output_path, **kwargs):
    spark = initiate_spark_session()
    data = spark.read.csv(input_path, header=True, inferSchema=True)
    data = data.withColumn('trestbps_sqrt', col('trestbps') ** 0.5)
    fe_path = output_path.replace('.csv', '_fe2.csv')
    data.write.csv(fe_path, header=True, mode='overwrite')
    logging.info(f"Saved feature engineered Spark data to {fe_path}")
    return fe_path

# Train SVM model using Spark
def spark_train_svm_model():
    spark = initiate_spark_session()
    data = spark.read.csv('/tmp/heart_disease_subset_fe1.csv', header=True, inferSchema=True)
    feature_cols = [col for col in data.columns if col != 'target']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    train_data, test_data = data.randomSplit([0.9, 0.1], seed=42)
    svm_model = LinearSVC(labelCol='target', featuresCol='features', maxIter=100)
    pipeline = Pipeline(stages=[assembler, svm_model])
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    print(f"SVM Model Accuracy: {accuracy:.4f}")

# Train Logistic Regression model using Spark
def spark_train_logistic_model():
    spark = initiate_spark_session()
    data = spark.read.csv('/tmp/heart_disease_subset_fe2.csv', header=True, inferSchema=True)
    feature_cols = [col for col in data.columns if col != 'target']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    train_data, test_data = data.randomSplit([0.9, 0.1], seed=42)
    logistic_model = SparkLogisticRegression(labelCol='target', featuresCol='features', maxIter=1000)
    pipeline = Pipeline(stages=[assembler, logistic_model])
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")

# Feature engineering strategies in pandas
def feature_eng_1(file_path, **kwargs):
    logging.info(f"Reading data from {file_path}")
    data = pd.read_csv(file_path)
    data['age_squared'] = data['age'] ** 2
    fe_path = file_path.replace('.csv', '_fe1.csv')
    data.to_csv(fe_path, index=False)
    logging.info(f"Saved feature engineered data to {fe_path}")
    return fe_path

def feature_eng_2(file_path, **kwargs):
    logging.info(f"Reading data from {file_path}")
    data = pd.read_csv(file_path)
    data['trestbps_sqrt'] = data['trestbps'] ** 0.5
    fe_path = file_path.replace('.csv', '_fe2.csv')
    data.to_csv(fe_path, index=False)
    logging.info(f"Saved feature engineered data to {fe_path}")
    return fe_path

# Train models using pandas and scikit-learn
def train_svm(file_path, **kwargs):
    logging.info(f"Reading data from {file_path}")
    data = pd.read_csv(file_path)
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = SVC(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"SVM Model Accuracy: {accuracy:.4f}")

def train_logistic(file_path, **kwargs):
    logging.info(f"Reading data from {file_path}")
    data = pd.read_csv(file_path)
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = SklearnLogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")

# Define Airflow tasks
load_data_task = PythonOperator(
    task_id='fetch_data_from_s3',
    python_callable=fetch_data_from_s3,
    dag=dag,
)

eda_task = PythonOperator(
    task_id='perform_eda',
    python_callable=perform_eda,
    provide_context=True,
    dag=dag,
)

spark_eda_task = PythonOperator(
    task_id='spark_eda',
    python_callable=spark_eda,
    op_kwargs={'input_path': '/tmp/heart_disease.csv', 'output_path': '/tmp/heart_disease_subset_eda.csv'},
    dag=dag,
)

fe1_task = PythonOperator(
    task_id='feature_eng_1',
    python_callable=feature_eng_1,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset.csv'},
    dag=dag,
)

fe2_task = PythonOperator(
    task_id='feature_eng_2',
    python_callable=feature_eng_2,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset.csv'},
    dag=dag,
)

spark_fe1_task = PythonOperator(
    task_id='spark_feature_eng_1',
    python_callable=spark_feature_eng_1,
    op_args=['/tmp/heart_disease_subset_eda.csv', '/tmp/heart_disease_subset_fe1.csv'],
    dag=dag,
)

spark_fe2_task = PythonOperator(
    task_id='spark_feature_eng_2',
    python_callable=spark_feature_eng_2,
    op_args=['/tmp/heart_disease_subset_eda.csv', '/tmp/heart_disease_subset_fe2.csv'],
    dag=dag,
)

train_svm_task = PythonOperator(
    task_id='train_svm',
    python_callable=train_svm,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset_fe1.csv'},
    dag=dag,
)

train_logistic_task = PythonOperator(
    task_id='train_logistic',
    python_callable=train_logistic,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset_fe2.csv'},
    dag=dag,
)

spark_train_svm_task = PythonOperator(
    task_id='spark_train_svm_model',
    python_callable=spark_train_svm_model,
    dag=dag,
)

spark_train_logistic_task = PythonOperator(
    task_id='spark_train_logistic_model',
    python_callable=spark_train_logistic_model,
    dag=dag,
)

# Define task dependencies
load_data_task >> [eda_task, spark_eda_task]

eda_task >> [fe1_task, fe2_task]
fe1_task >> train_svm_task
fe2_task >> train_logistic_task

spark_eda_task >> [spark_fe1_task, spark_fe2_task]
spark_fe1_task >> spark_train_svm_task
spark_fe2_task >> spark_train_logistic_task
