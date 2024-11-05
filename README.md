# 🛒 Online Sales Dataset Analysis & Model Training

Welcome to the **Online Sales Dataset** project! This project is aimed at predicting customer satisfaction using an online sales dataset. This README file provides a step-by-step guide to the data preprocessing, training, and saving a decision tree model for classifying customer satisfaction.

---

## 📂 Dataset Overview

The **Online Sales Dataset** contains data on various online transactions, covering details about:
- Customer IDs
- Product Categories
- Payment Methods
- Sales Channels
- Shipping Providers
- Warehouse Locations

The **objective** is to predict whether a customer is satisfied (`1`) or dissatisfied (`0`) based on transaction details.

---

## 🔧 Requirements

Install the following Python packages to run the project:
```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## ⚙️ Project Structure

- **online_sales_dataset.csv**: The raw data file.
- **3.0-decision-tree-model-training.py**: The main training script for data processing, model training, and visualization.
- **README.md**: Documentation of the project.
- **Output Files**:
  - **online_sales_model.pkl**: Trained Decision Tree model file.
  - **label_encoders.pkl**: Label encoders for categorical data.
  - **scaler.pkl**: Scaler for feature normalization.
  - **decision_tree.png**: Image of the trained decision tree model.

---

## 📝 Steps to Run the Project

### 1. 📥 Load the Dataset
In the training script (`3.0-decision-tree-model-training.py`):
```python
df = pd.read_csv("C:/Users/HP/OneDrive/Desktop/Online sales 1/DATA1/online_sales_dataset.csv")
```

### 2. 🧹 Clean the Data
Remove rows with missing values in critical columns:
```python
df = df.dropna(subset=['CustomerID', 'ShippingCost'])
```

### 3. 🔄 Encode Categorical Variables
Convert categorical columns to numerical values using `LabelEncoder`:
```python
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for column in ['Country', 'PaymentMethod', 'Category', 'SalesChannel', 'ReturnStatus', 'ShipmentProvider', 'WarehouseLocation', 'OrderPriority']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
```

### 4. 🎯 Define the Target Variable
Create a binary target column, `satisfied`, based on `ReturnStatus`:
```python
df['satisfied'] = np.where(df['ReturnStatus'] == label_encoders['ReturnStatus'].transform(['Not Returned'])[0], 1, 0)
```

### 5. 🚫 Drop Unnecessary Columns
Remove columns irrelevant to the model:
```python
df = df.drop(columns=['InvoiceNo', 'StockCode', 'Description', 'InvoiceDate', 'ReturnStatus'])
```

### 6. 📊 Split Features & Target
Separate the features (`X`) and the target variable (`y`):
```python
X = df.drop(columns=['satisfied'])
y = df['satisfied']
```

### 7. 🔍 Scale the Features
Standardize the feature values for better model performance:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 8. ✂️ Split the Dataset
Split data into training and test sets:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### 9. 🌳 Train a Decision Tree Model
Train a Decision Tree classifier:
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

### 10. 📈 Evaluate the Model
Generate predictions and print the classification report:
```python
from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=1))
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
```

### 11. 💾 Save the Model, Encoders, and Scaler
Store the model and transformations for future use:
```python
import pickle

# Save model
with open('C:/Users/HP/OneDrive/Desktop/online_sales_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save label encoders
with open('C:/Users/HP/OneDrive/Desktop/label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

# Save scaler
with open('C:/Users/HP/OneDrive/Desktop/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
```

### 12. 🖼️ Visualize the Decision Tree
Save an image of the decision tree structure:
```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Dissatisfied', 'Satisfied'], rounded=True)
plt.savefig('C:/Users/HP/OneDrive/Desktop/decision_tree.png')
plt.close()


## 📊 Results

The classification report and accuracy are printed after model evaluation. The trained model, label encoders, scaler, and decision tree image are saved in the specified paths.

## 🎉 Conclusion

This project demonstrates the end-to-end process of loading, processing, and modeling customer satisfaction prediction using online sales data. With the decision tree model, 
users can predict satisfaction outcomes based on transaction details.

Enjoy exploring and improving the **Online Sales Dataset** project! 🚀
