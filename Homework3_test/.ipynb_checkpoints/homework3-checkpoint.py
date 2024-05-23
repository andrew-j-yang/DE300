from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean as _mean, lit
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

df = spark.read.csv("s3://andrewawsbucket/heart_disease.csv", header = True, inferSchema = True)
df = df.drop(df.tail(2))

# Selecting only the desired columns
selected_columns = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
df = df.select(selected_columns)

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
    mode_value = df.groupBy(column).count().orderBy('count', ascending=False).first()[0]
    df = df.withColumn(column, when((col(column) != 0) & (col(column) != 1) | col(column).isNull(), mode_value).otherwise(col(column)))

# Replace missing values with average values
for col_name in ['thaldur', 'thalach', 'trestbps']:
    mean_value = df.select(_mean(col(col_name))).collect()[0][0]
    df = df.withColumn(col_name, when(col(col_name).isNull(), mean_value).otherwise(col(col_name)))

# Calculate the average of the 'oldpeak' column
average_oldpeak = df.select(_mean(col('oldpeak'))).collect()[0][0]
df = df.withColumn('oldpeak', when((col('oldpeak') < 0) | (col('oldpeak') > 4) | col('oldpeak').isNull(), average_oldpeak).otherwise(col('oldpeak')))

# Replace missing values, values less than 0, and values greater than 4 with the average
df['oldpeak'] = df['oldpeak'].apply(lambda x: average_oldpeak if pd.isnull(x) or x < 0 or x > 4 else x)

valid_categories = {
    'cp': {1, 2, 3, 4},
    'slope': {1,2,3},
}

for column, valid_set in valid_categories.items():
    mode_value = df.groupBy(column).count().orderBy('count', ascending=False).first()[0]
    df = df.withColumn(column, when(~col(column).isin(valid_set), mode_value).otherwise(col(column)))

# Set the webpage URL for fetching data
data_url = "https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking-and-vaping/latest-release"
# Send a GET request to the URL
web_response = requests.get(data_url)
# Parse the HTML content using BeautifulSoup
html_content = BeautifulSoup(web_response.content, 'html.parser')
# Specify a key phrase from the chart caption to locate the right data
search_caption = "Proportion of people 15 years and over who were current daily smokers by age, 2011"  # update this as needed
# Initialize variable to store the desired div
target_div = None

# Loop through all div elements with the specified class
for container in html_content.find_all('div', {'class': 'chart-data-wrapper'}):
    # Extract the caption text
    chart_caption = container.find('pre', {'class': 'chart-caption'}).text
    # Check if the specified caption part is in the extracted caption
    if search_caption in chart_caption:
        target_div = container
        break
# Parse and extract chart data from JSON format
chart_data = json.loads(target_div.find('pre', {'class': 'chart-data'}).text)


desired_values = chart_data[7]


# Smoking rates by age group as extracted
smoking_rates = [item for sublist in desired_values for item in sublist]

# Define age bins corresponding to the age groups in the rate table
bins = [0, 17, 24, 34, 44, 54, 64, 74, 120]

# Assign each age to an age group
df = df.withColumn('age_group_ABS', when((col('age') < 18), 0)
                   .when((col('age') < 25), 1)
                   .when((col('age') < 35), 2)
                   .when((col('age') < 45), 3)
                   .when((col('age') < 55), 4)
                   .when((col('age') < 65), 5)
                   .when((col('age') < 75), 6)
                   .otherwise(7))

# Function to impute NaN based on smoking probability
def impute_smoking(smoke, age_group_ABS):
    if smoke is None:
        rate = smoking_rates[age_group_ABS]
        return 1 if np.random.rand() < rate else 0
    return smoke

df['ABS smoke'] = df.apply(impute_smoking, axis=1)


# Define the URL containing information on smoking statistics for various demographics.
source_url = "https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm"

# Fetch the data from the source.
server_response = requests.get(source_url)
if server_response.status_code != 200:
    print("Failed to retrieve data")

# Retrieve and parse the HTML content from the fetched data.
html_data = server_response.content
selector = Selector(text=html_data)
target_div = selector.xpath("//div[@class='row '][3]")

# Extracting lists from specific parts of the webpage.
list_selector = target_div.xpath("//ul[@class='block-list']")

# Extracting textual content from the lists related to gender smoking rates.
gender_data = list_selector[0].xpath(".//li/text()")
age_data = list_selector[1].xpath(".//li/text()")

# Parsing percentages for male and female from the extracted text.
male_rate = float(gender_data[0].extract().split("(")[1].split("%)")[0])
female_rate = float(gender_data[1].extract().split("(")[1].split("%)")[0])

# Creating a dictionary to store age-specific smoking rates.
age_rates = {}
for item in age_data:
    age_range = item.extract().split("aged ")[1].split(" years")[0]
    rate = float(item.extract().split("(")[1].split("%)")[0])
    if "–" in age_range:
        age_limits = age_range.split("–")
        age_rates[(int(age_limits[0]), int(age_limits[1]))] = rate
    else:
        age_rates[(int(age_range), float('inf'))] = rate

# Adjust male smoking rates based on comparison with female rates.
adjusted_male_rates = {key: value * (male_rate / female_rate) for key, value in age_rates.items()}

print("Adjusted male smoking rates by age:", adjusted_male_rates)
print("Female smoking rates by age:", age_rates)


bins = [18, 24, 44, 64, float('inf')]
labels = [(18, 24), (25, 44), (45, 64), (65, float('inf'))]

df['age_group_CDC'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

def impute_smoking(row):
    if pd.isna(row['smoke']):
        age_group = row['age_group_CDC']
        if row['sex'] == 1:  # Male
            return 1 if np.random.rand() < adjusted_male_rates[age_group] / 100 else 0
        else:  # Female
            return 1 if np.random.rand() < age_rates[age_group] / 100 else 0
    else:
        return row['smoke']

df['CDC smoke'] = df.apply(impute_smoking, axis=1)


# Define the function to impute the 'smoke' column
def impute_smoke(row):
    if pd.isna(row['smoke']):
        if row['ABS smoke'] == 0 and row['CDC smoke'] == 0:
            return 0
        else:
            return 1
    else:
        return row['smoke']

# Apply the function across each row
df['smoke'] = df.apply(impute_smoke, axis=1)


df = df.drop('age_group_ABS', axis=1)
df = df.drop('age_group_CDC', axis=1)

from sklearn.model_selection import train_test_split

# Assume df is your DataFrame and 'target' is the column with 0 or 1
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target
print(X)

# Splitting the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
nan_counts = X.isna().sum()

# This will give you a Series with the count of NaNs in each column
print(nan_counts)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Setting up the logistic regression with hyperparameter grid
log_reg_params = {'C': [0.01, 0.1, 1, 10, 100]}
log_reg = LogisticRegression(solver='liblinear')

log_reg_grid = GridSearchCV(log_reg, log_reg_params, cv=5, scoring='accuracy')
log_reg_grid.fit(X_train, y_train)
print("Best parameters for Logistic Regression:", log_reg_grid.best_params_)
print("Cross-validated accuracy:", log_reg_grid.best_score_)

# Setting up the random forest classifier with hyperparameter grid
rf_params = {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30]}
rf = RandomForestClassifier()

rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)
print("Best parameters for Random Forest:", rf_grid.best_params_)
print("Cross-validated accuracy:", rf_grid.best_score_)

# Compare the performance and select the best model
if log_reg_grid.best_score_ > rf_grid.best_score_:
    final_model = log_reg_grid.best_estimator_
    print("Selected Logistic Regression as the final model.")
else:
    final_model = rf_grid.best_estimator_
    print("Selected Random Forest as the final model.")

# Final evaluation on the test data
y_pred = final_model.predict(X_test)
print("Performance on the test set:")
print(classification_report(y_test, y_pred))







