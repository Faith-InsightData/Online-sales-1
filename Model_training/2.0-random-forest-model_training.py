import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load the dataset
df = pd.read_csv("C:/Users/HP/OneDrive/Desktop/Online sales 1/DATA1/online_sales_dataset.csv")

# Drop rows with missing values in 'CustomerID' and 'ShippingCost'
df = df.dropna(subset=['CustomerID', 'ShippingCost'])

# Encode categorical columns using LabelEncoder
label_encoders = {}
for column in ['Country', 'PaymentMethod', 'Category', 'SalesChannel', 'ReturnStatus', 'ShipmentProvider', 'WarehouseLocation', 'OrderPriority']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define the target variable
# Assuming 'ReturnStatus' indicates dissatisfaction if 'Returned'
df['satisfied'] = np.where(df['ReturnStatus'] == label_encoders['ReturnStatus'].transform(['Not Returned'])[0], 1, 0)

# Drop unnecessary columns for modeling
df = df.drop(columns=['InvoiceNo', 'StockCode', 'Description', 'InvoiceDate', 'ReturnStatus'])

# Split data into features and target
X = df.drop(columns=['satisfied'])
y = df['satisfied']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a logistic regression model with increased max_iter
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Predict on the test set and evaluate
y_pred = model.predict(X_test)
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=1))
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

# Save the trained model as online_sales_model.pkl
with open('C:/Users/HP/OneDrive/Desktop/online_sales_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save label encoders for categorical columns
with open('C:/Users/HP/OneDrive/Desktop/label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

# Save the scaler for feature scaling
with open('C:/Users/HP/OneDrive/Desktop/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
