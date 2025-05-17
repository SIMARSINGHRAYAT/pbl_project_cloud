from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

disease_models = {
    'diabetes': {
        'features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        'model_path': 'models/diabetes_model.pkl',
        'scaler_path': 'models/diabetes_scaler.pkl',
        'data_path': 'data/diabetes.csv',
        'target': 'Outcome'
    },
    'heart': {
        'features': ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'],
        'model_path': 'models/heart_model.pkl',
        'scaler_path': 'models/heart_scaler.pkl',
        'data_path': 'data/heart.csv',
        'target': 'HeartDisease'
    },
    'liver': {
        'features': ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
                    'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
                    'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 
                    'Albumin_and_Globulin_Ratio'],
        'model_path': 'models/liver_model.pkl',
        'scaler_path': 'models/liver_scaler.pkl',
        'data_path': 'data/liver.csv',
        'target': 'Dataset'
    }
}

def create_sample_data():
    np.random.seed(42)
    
    diabetes_data = {
        'Pregnancies': np.random.randint(0, 17, 1000),
        'Glucose': np.random.randint(50, 200, 1000),
        'BloodPressure': np.random.randint(40, 130, 1000),
        'SkinThickness': np.random.randint(0, 100, 1000),
        'Insulin': np.random.randint(0, 850, 1000),
        'BMI': np.random.uniform(15, 60, 1000),
        'DiabetesPedigreeFunction': np.random.uniform(0.05, 2.5, 1000),
        'Age': np.random.randint(20, 90, 1000)
    }
    
    df_diabetes = pd.DataFrame(diabetes_data)
    df_diabetes['Outcome'] = ((df_diabetes['Glucose'] > 140) & 
                             (df_diabetes['BMI'] > 30)).astype(int)
    df_diabetes.to_csv('data/diabetes.csv', index=False)
    
    heart_data = {
        'Age': np.random.randint(20, 90, 1000),
        'Sex': np.random.randint(0, 2, 1000),
        'ChestPainType': np.random.randint(0, 4, 1000),
        'RestingBP': np.random.randint(90, 200, 1000),
        'Cholesterol': np.random.randint(100, 600, 1000),
        'FastingBS': np.random.randint(0, 2, 1000),
        'RestingECG': np.random.randint(0, 3, 1000),
        'MaxHR': np.random.randint(60, 220, 1000),
        'ExerciseAngina': np.random.randint(0, 2, 1000),
        'Oldpeak': np.random.uniform(0, 7, 1000),
        'ST_Slope': np.random.randint(0, 3, 1000)
    }
    
    df_heart = pd.DataFrame(heart_data)
    df_heart['HeartDisease'] = ((df_heart['RestingBP'] > 140) & 
                               (df_heart['Cholesterol'] > 240) &
                               (df_heart['Age'] > 50)).astype(int)
    df_heart.to_csv('data/heart.csv', index=False)
    
    liver_data = {
        'Age': np.random.randint(20, 90, 1000),
        'Gender': np.random.randint(0, 2, 1000),
        'Total_Bilirubin': np.random.uniform(0.1, 8, 1000),
        'Direct_Bilirubin': np.random.uniform(0.05, 3, 1000),
        'Alkaline_Phosphotase': np.random.randint(50, 500, 1000),
        'Alamine_Aminotransferase': np.random.randint(10, 300, 1000),
        'Aspartate_Aminotransferase': np.random.randint(10, 300, 1000),
        'Total_Protiens': np.random.uniform(3, 9, 1000),
        'Albumin': np.random.uniform(1, 6, 1000),
        'Albumin_and_Globulin_Ratio': np.random.uniform(0.3, 3, 1000)
    }
    
    df_liver = pd.DataFrame(liver_data)
    df_liver['Dataset'] = ((df_liver['Total_Bilirubin'] > 1.2) & 
                          (df_liver['Alamine_Aminotransferase'] > 50) &
                          (df_liver['Aspartate_Aminotransferase'] > 40)).astype(int)
    df_liver.to_csv('data/liver.csv', index=False)

def preprocess_data(df, disease_info):
    features = disease_info['features']
    target = disease_info['target']
    
    X = df[features].copy()
    y = df[target]
    
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    imputer = SimpleImputer(strategy='median')
    X[numerical_features] = imputer.fit_transform(X[numerical_features])
    
    for col in numerical_features:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X[col] = np.where(X[col] > upper_bound, upper_bound, 
                          np.where(X[col] < lower_bound, lower_bound, X[col]))
    
    return X, y

def train_model(disease_name):
    try:
        if disease_name not in disease_models:
            return {"error": f"Disease {disease_name} is not supported"}
        
        disease_info = disease_models[disease_name]
        
        if not os.path.exists(disease_info['data_path']):
            create_sample_data()
        
        df = pd.read_csv(disease_info['data_path'])
        X, y = preprocess_data(df, disease_info)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        best_accuracy = 0
        best_model = None
        model_results = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            model_results[name] = float(accuracy)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        
        with open(disease_info['model_path'], 'wb') as f:
            pickle.dump(best_model, f)
        
        with open(disease_info['scaler_path'], 'wb') as f:
            pickle.dump(scaler, f)
        
        return {
            "success": True,
            "message": f"Model for {disease_name} trained successfully",
            "best_model": max(model_results.items(), key=lambda x: x[1])[0],
            "accuracy": float(best_accuracy),
            "model_comparison": model_results
        }
    
    except Exception as e:
        return {"error": str(e)}

def predict_disease(disease_name, input_data):
    try:
        if disease_name not in disease_models:
            return {"error": f"Disease {disease_name} is not supported"}
        
        disease_info = disease_models[disease_name]
        
        if not os.path.exists(disease_info['model_path']) or not os.path.exists(disease_info['scaler_path']):
            train_result = train_model(disease_name)
            if "error" in train_result:
                return train_result
        
        with open(disease_info['model_path'], 'rb') as f:
            model = pickle.load(f)
        
        with open(disease_info['scaler_path'], 'rb') as f:
            scaler = pickle.load(f)
        
        input_df = pd.DataFrame([input_data])
        
        for feature in disease_info['features']:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        input_df = input_df[disease_info['features']]
        
        numerical_features = input_df.select_dtypes(include=['int64', 'float64']).columns
        imputer = SimpleImputer(strategy='median')
        input_df[numerical_features] = imputer.fit_transform(input_df[numerical_features])
        
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "disease": disease_name,
            "result": "Positive" if prediction == 1 else "Negative"
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train_endpoint():
    data = request.json
    disease_name = data.get('disease')
    
    if not disease_name:
        return jsonify({"error": "Disease name is required"}), 400
    
    result = train_model(disease_name)
    return jsonify(result)

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    data = request.json
    disease_name = data.get('disease')
    input_data = data.get('input_data')
    
    if not disease_name or not input_data:
        return jsonify({"error": "Disease name and input data are required"}), 400
    
    result = predict_disease(disease_name, input_data)
    return jsonify(result)

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    return jsonify({
        "diseases": list(disease_models.keys()),
        "features": {disease: info['features'] for disease, info in disease_models.items()}
    })

@app.route('/api/add_disease', methods=['POST'])
def add_disease():
    data = request.json
    disease_name = data.get('disease_name')
    features = data.get('features')
    target = data.get('target', 'Target')
    
    if not disease_name or not features:
        return jsonify({"error": "Disease name and features are required"}), 400
    
    disease_models[disease_name] = {
        'features': features,
        'model_path': f'models/{disease_name}_model.pkl',
        'scaler_path': f'models/{disease_name}_scaler.pkl',
        'data_path': f'data/{disease_name}.csv',
        'target': target
    }
    
    return jsonify({"success": True, "message": f"Disease {disease_name} added successfully"})

@app.route('/api/sample_data/<disease>', methods=['GET'])
def get_sample_data(disease):
    if disease not in disease_models:
        return jsonify({"error": f"Disease {disease} is not supported"}), 400
    
    disease_info = disease_models[disease]
    
    if not os.path.exists(disease_info['data_path']):
        create_sample_data()
    
    df = pd.read_csv(disease_info['data_path'])
    return jsonify(df.head(5).to_dict(orient='records'))

if __name__ == '__main__':
    create_sample_data()
    
    for disease in disease_models:
        train_model(disease)
    
    app.run(host='0.0.0.0', port=5000)