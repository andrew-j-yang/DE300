from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean as _mean, lit
from pyspark.sql.functions import col, lit, row_number, rand, when, isnan, count, udf, sum
from pyspark.sql.types import IntegerType, FloatType, DoubleType, LongType, StringType
from pyspark.ml.feature import Imputer, VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors, DenseVector
from pyspark.ml import Pipeline
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import MulticlassMetrics

import numpy as np
import requests
from bs4 import BeautifulSoup
import json
from scrapy import Selector

def main():
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("Heart Disease Analysis") \
        .getOrCreate()
    
    df = spark.read.csv("s3://andrewawsbucket/heart_disease.csv", header = True, inferSchema = True)
    
    # Get the number of rows in the DataFrame
    row_count = df.count()
    
    # Calculate the number of rows to keep (excluding the last two)
    rows_to_keep = row_count - 2
    
    # Select all rows except the last two
    df = df.limit(rows_to_keep)
    
    
    
    
    # Selecting only the desired columns
    selected_columns = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    
    # Selecting only the desired columns
    df = df.select(*selected_columns)
    
    
    
    
    
    binary_attributes = [
        'painloc',      # Chest pain location (1 = substernal, 0 = otherwise)
        'painexer',     # Whether pain is provoked by exertion (1 = yes, 0 = no)
        'fbs',          # Fasting blood sugar > 120 mg/dL (1 = true, 0 = false)
        'prop',         # Beta blocker used during exercise ECG (1 = yes, 0 = no)
        'nitr',         # Nitrates used during exercise ECG (1 = yes, 0 = no)
        'pro',          # Calcium channel blocker used during exercise ECG (1 = yes, 0 = no)
        'diuretic',     # Diuretic used during exercise ECG (1 = yes, 0 = no)
        'exang'
    ]
    
    for column in binary_attributes:
        # Calculate the mode of the binary column
        mode_value = df.groupBy(column).count().orderBy('count', ascending=False).first()[0]
        
        # Replace non-binary and NaN values with the mode
        df = df.withColumn(column, when((col(column).isNull()) | (~col(column).isin(0, 1)), mode_value).otherwise(col(column)))
    
    # Replace missing values in 'thaldur' column with the average of the column
    thaldur_average = df.agg(_mean(col('thaldur')).alias('mean')).first()['mean']
    df = df.withColumn('thaldur', when(col('thaldur').isNull(), thaldur_average).otherwise(col('thaldur')))
    
    # Replace missing values in 'thalach' column with the average of the column
    thalach_average = df.agg(_mean(col('thalach')).alias('mean')).first()['mean']
    df = df.withColumn('thalach', when(col('thalach').isNull(), thalach_average).otherwise(col('thalach')))
    
    # Replace missing values in 'trestbps' column with the average of the column
    trestbps_average = df.agg(_mean(col('trestbps')).alias('mean')).first()['mean']
    df = df.withColumn('trestbps', when(col('trestbps').isNull(), trestbps_average).otherwise(col('trestbps')))
    
    # Calculate the average of the 'oldpeak' column
    average_oldpeak = df.agg(_mean(col('oldpeak')).alias('mean')).first()['mean']
    
    # Replace missing values, values less than 0, and values greater than 4 with the average
    df = df.withColumn('oldpeak', when(col('oldpeak').isNull() | (col('oldpeak') < 0) | (col('oldpeak') > 4), average_oldpeak).otherwise(col('oldpeak')))
    
    valid_categories = {
        'cp': {1, 2, 3, 4},
        'slope': {1, 2, 3},
    }
    
    # for column, valid_set in valid_categories.items():
    #     mode_value = df.groupBy(column).count().orderBy('count', ascending=False).first()[0]
    #     df = df.withColumn(column, when(~col(column).isin(valid_set), mode_value).otherwise(col(column)))
    
    # Function to impute invalid values with mode
    def impute_with_mode(df, column, valid_set):
        mode_value = df.groupBy(column).count().orderBy('count', ascending=False).first()[0]
        df = df.withColumn(column, when(~col(column).isin(valid_set) | col(column).isNull(), mode_value).otherwise(col(column)))
        return df
    
    # Apply the imputation to each column
    for column, valid_set in valid_categories.items():
        df = impute_with_mode(df, column, valid_set)
    
    
    
    #Fill the 'smoke' column with 0s and 1s based on a random number generator
    df = df.withColumn('smoke', when(rand() > 0.5, 1).otherwise(0))
    
    
    
    
    # # Split the data into features and target
    # target_column = 'target'
    # feature_columns = [column for column in df.columns if column != target_column]
    
    # # Split the data with stratification
    # stratified_df = df.withColumn('rand', rand())
    # train_df = stratified_df.where(col('rand') >= 0.1).drop('rand')
    # test_df = stratified_df.where(col('rand') < 0.1).drop('rand')
    
    # # Count NaNs in each column
    # nan_counts = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
    
    # # Setting up the logistic regression with hyperparameter grid
    # log_reg = LogisticRegression(labelCol=target_column)
    # log_reg_param_grid = ParamGridBuilder() \
    #     .addGrid(log_reg.regParam, [0.01, 0.1, 1, 10, 100]) \
    #     .build()
    
    # # Setting up cross-validation
    # crossval_log_reg = CrossValidator(estimator=log_reg,
    #                                   estimatorParamMaps=log_reg_param_grid,
    #                                   evaluator=MulticlassClassificationEvaluator(labelCol=target_column, metricName='accuracy'),
    #                                   numFolds=5)
    
    # # Fit logistic regression model
    # log_reg_model = crossval_log_reg.fit(train_df)
    # best_log_reg_model = log_reg_model.bestModel
    
    # print("Best parameters for Logistic Regression:", best_log_reg_model.extractParamMap())
    # print("Cross-validated accuracy:", log_reg_model.avgMetrics[0])
    
    # # Setting up the random forest classifier with hyperparameter grid
    # rf = RandomForestClassifier(labelCol=target_column)
    # rf_param_grid = ParamGridBuilder() \
    #     .addGrid(rf.numTrees, [10, 50, 100, 200]) \
    #     .addGrid(rf.maxDepth, [5, 10, 20, 30]) \
    #     .build()
    
    # # Setting up cross-validation
    # crossval_rf = CrossValidator(estimator=rf,
    #                              estimatorParamMaps=rf_param_grid,
    #                              evaluator=MulticlassClassificationEvaluator(labelCol=target_column, metricName='accuracy'),
    #                              numFolds=5)
    
    # # Fit random forest model
    # rf_model = crossval_rf.fit(train_df)
    # best_rf_model = rf_model.bestModel
    
    # print("Best parameters for Random Forest:", best_rf_model.extractParamMap())
    # print("Cross-validated accuracy:", rf_model.avgMetrics[0])
    
    # # Compare the performance and select the best model
    # if log_reg_model.avgMetrics[0] > rf_model.avgMetrics[0]:
    #     final_model = best_log_reg_model
    #     print("Selected Logistic Regression as the final model.")
    # else:
    #     final_model = best_rf_model
    #     print("Selected Random Forest as the final model.")
    
    # # Final evaluation on the test data
    # predictions = final_model.transform(test_df)
    # evaluator = MulticlassClassificationEvaluator(labelCol=target_column, metricName='accuracy')
    # accuracy = evaluator.evaluate(predictions)
    
    # print("Performance on the test set:")
    # print("Accuracy:", accuracy)



    # Assuming df is your DataFrame and selected_columns are defined
    feature_columns = selected_columns
    
    # Assemble features
    vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = vector_assembler.transform(df)
    
    # Rename the target column to label
    df = df.withColumnRenamed("target", "label")
    
    # Split the DataFrame into training and testing sets
    train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)
    
    def evaluate_model(predictions):
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        return evaluator.evaluate(predictions)
    
    # Train and evaluate a logistic regression model
    log_reg = LogisticRegression(labelCol='label', featuresCol='features')
    log_reg_model = log_reg.fit(train_df)
    log_reg_predictions = log_reg_model.transform(test_df)
    log_reg_accuracy = evaluate_model(log_reg_predictions)
    
    # Train and evaluate a decision tree model
    decision_tree = DecisionTreeClassifier(labelCol='label', featuresCol='features')
    decision_tree_model = decision_tree.fit(train_df)
    decision_tree_predictions = decision_tree_model.transform(test_df)
    decision_tree_accuracy = evaluate_model(decision_tree_predictions)
    
    # Train and evaluate a random forest model
    random_forest = RandomForestClassifier(labelCol='label', featuresCol='features')
    random_forest_model = random_forest.fit(train_df)
    random_forest_predictions = random_forest_model.transform(test_df)
    random_forest_accuracy = evaluate_model(random_forest_predictions)
    
    # Determine the best model based on accuracy
    accuracies = {
        "Logistic Regression": log_reg_accuracy,
        "Decision Tree": decision_tree_accuracy,
        "Random Forest": random_forest_accuracy
    }
    best_model_name, best_model_accuracy = max(accuracies.items(), key=lambda item: item[1])
    
    print(f"The best model is {best_model_name} with an accuracy of {best_model_accuracy:.2f}")
    
    # Select the best model and evaluate it on the test set
    best_model = log_reg_model if best_model_name == "Logistic Regression" else \
                 decision_tree_model if best_model_name == "Decision Tree" else \
                 random_forest_model
    
    best_model_predictions = best_model.transform(test_df)
    test_set_accuracy = evaluate_model(best_model_predictions)
    print(f"Test set accuracy: {test_set_accuracy:.2f}")
    
    # Generate classification report
    predictions_and_labels = best_model_predictions.select("prediction", "label").rdd.map(lambda row: (float(row[0]), float(row[1])))
    metrics = MulticlassMetrics(predictions_and_labels)
    
    print("Classification report on test set:")
    print(metrics.confusionMatrix().toArray())


    spark.stop()

main()




