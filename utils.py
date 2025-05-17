import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(file_path, target_column, categorical_features=None, numerical_features=None):
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        return None, None, None
    
    if categorical_features is None:
        categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column in categorical_features:
            categorical_features.remove(target_column)
    
    if numerical_features is None:
        numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target_column in numerical_features:
            numerical_features.remove(target_column)
    
    y = data[target_column]
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X = data.drop(target_column, axis=1)
    X_processed = preprocessor.fit_transform(X)
    
    return X, y, preprocessor

def handle_outliers(df, method='iqr', columns=None):
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean[col] = np.where(df[col] > upper_bound, upper_bound,
                                    np.where(df[col] < lower_bound, lower_bound, df[col]))
        
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            threshold = 3
            
            df_clean[col] = np.where(abs(df[col] - mean) > threshold * std,
                                    np.sign(df[col] - mean) * threshold * std + mean,
                                    df[col])
    
    return df_clean

def evaluate_model(model, X_test, y_test, X_train=None, y_train=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except:
            metrics['roc_auc'] = None
    
    if X_train is not None and y_train is not None:
        y_train_pred = model.predict(X_train)
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
    
    return metrics