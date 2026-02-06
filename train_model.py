import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_save_model():
    print("Loading data...")
    # Load dataset
    try:
        data = pd.read_csv('Student Depression Dataset.csv')
    except FileNotFoundError:
        print("Error: 'Student Depression Dataset.csv' not found.")
        return

    # Drop id column
    if 'id' in data.columns:
        data.drop('id', axis=1, inplace=True)

    print("Preprocessing data...")
    # Handle missing values
    data_num = data.select_dtypes(include='number')
    data_cat = data.select_dtypes(include='object')

    data_num.fillna(data_num.mean(), inplace=True)
    data_cat.fillna(data_cat.mode().iloc[0], inplace=True) # Use iloc[0] for mode to get a specific value

    data = pd.concat([data_num, data_cat], axis=1)

    # Encode categorical variables
    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    # Split data
    X = data.drop('Depression', axis=1)
    y = data['Depression']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest model...")
    # Train Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate
    y_pred = rf_model.predict(X_test)
    print(f'Model Accuracy: {accuracy_score(y_test, y_pred):.4f}')

    # Save artifacts
    artifacts = {
        'model': rf_model,
        'encoders': label_encoders,
        'features': X.columns.tolist()
    }
    
    output_file = 'depression_model.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(artifacts, f)
    
    print(f"Model and artifacts saved to {output_file}")

if __name__ == "__main__":
    train_and_save_model()
