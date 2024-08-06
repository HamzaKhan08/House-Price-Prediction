import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Loading the dataset
data = pd.read_csv('/Users/hamzakhan/Downloads/House Price Prediction By Hk/data.csv', header=0) 
# kindly change the path based on your directory for running this project

# Display the first few rows of dataset
print("Initial dataset:")
print(data.head())

# Checking for missing values in the dataset
print("\nMissing values in dataset:")
print(data.isnull().sum())

# Visualize the distribution of the target variable (SalePrice)
sns.histplot(data['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

# Visualize the correlation matrix
corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Split the data into train, validation, and test sets
train_data, temp_test_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_test_data, test_size=0.5, random_state=42)

# Save the datasets to CSV files
train_data.to_csv('/Users/hamzakhan/Downloads/House Price Prediction By Hk/train_data.csv', index=False)
val_data.to_csv('/Users/hamzakhan/Downloads/House Price Prediction By Hk/val_data.csv', index=False)
test_data.to_csv('/Users/hamzakhan/Downloads/House Price Prediction By Hk/test_data.csv', index=False)

print("\nDatasets created and saved as train_data.csv, val_data.csv, and test_data.csv.")

# Load the training data
X_train = train_data.drop('SalePrice', axis=1)
y_train = train_data['SalePrice']

# Load the validation data
X_val = val_data.drop('SalePrice', axis=1)
y_val = val_data['SalePrice']

# Load the test data (without target variable)
X_test = test_data.drop('SalePrice', axis=1)

# Select categorical and numerical columns
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == 'object']
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create the pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Fit the model on the training data
clf.fit(X_train, y_train)

# Predict on the validation data
y_pred = clf.predict(X_val)

# Calculate mean absolute error
mae = mean_absolute_error(y_val, y_pred)
print('\nMean Absolute Error on validation data:', mae)

# Fit the model on the entire training data (train + validation)
X_full_train = pd.concat([X_train, X_val])
y_full_train = pd.concat([y_train, y_val])
clf.fit(X_full_train, y_full_train)

# Preprocess the test data and make predictions
test_preds = clf.predict(X_test)

# Prepare the submission file
output = pd.DataFrame({'Id': test_data.index, 'SalePrice': test_preds})
output.to_csv('/Users/hamzakhan/Downloads/House Price Prediction By Hk/submission.csv', index=False)
print("\nPredictions saved to submission.csv")
