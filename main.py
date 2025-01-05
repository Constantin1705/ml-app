import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle
import os
import argparse

class MLApp:
    def __init__(self, random_state=42, cv_folds=5):
        self.random_state = random_state
        self.cv_folds = cv_folds

    def load_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")

    def preprocess(self, data, target_column):
        if target_column not in data.columns:
            raise ValueError("Specified target column not found in the dataset.")

        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Separate numerical and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        # Pipelines for preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        return preprocessor, X, y

    def create_pipeline(self, model, preprocessor):
        return Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

    def evaluate_model(self, pipeline, X, y, task_type):
        if task_type == 'classification':
            scoring = 'accuracy'
        elif task_type == 'regression':
            scoring = 'neg_mean_squared_error'
        else:
            raise ValueError("Invalid task type. Choose 'classification' or 'regression'.")

        scores = cross_val_score(pipeline, X, y, cv=self.cv_folds, scoring=scoring)
        if scoring == 'neg_mean_squared_error':
            scores = -scores  # Convert to positive MSE
        return scores

    def save_model(self, pipeline, file_name, folder_name):
        folder_path = os.path.join("models", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(pipeline, f)

    def save_report(self, results, file_name, folder_name):
        folder_path = os.path.join("reports", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w') as f:
            for model, metrics in results.items():
                f.write(f"{model}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value}\n")
                f.write("\n")

    def run(self, file_path, target_column, task_type):
        data = self.load_data(file_path)
        preprocessor, X, y = self.preprocess(data, target_column)

        file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]

        if task_type == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(random_state=self.random_state),
                'Random Forest Classifier': RandomForestClassifier(random_state=self.random_state),
                'Decision Tree Classifier': DecisionTreeClassifier(random_state=self.random_state)
            }
        elif task_type == 'regression':
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest Regressor': RandomForestRegressor(random_state=self.random_state),
                'Decision Tree Regressor': DecisionTreeRegressor(random_state=self.random_state)
            }
        else:
            raise ValueError("Invalid task type. Choose 'classification' or 'regression'.")

        results = {}
        for name, model in models.items():
            pipeline = self.create_pipeline(model, preprocessor)
            scores = self.evaluate_model(pipeline, X, y, task_type)
            results[name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            }
            model_file = f"{name.replace(' ', '_').lower()}_{task_type}.pkl"
            self.save_model(pipeline, model_file, file_name_without_extension)

        # Save results to a report
        report_file = f"{task_type}_results.txt"
        self.save_report(results, report_file, file_name_without_extension)

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML Pipeline Application")
    parser.add_argument("--file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--task", type=str, required=True, choices=['classification', 'regression'], help="Task type: classification or regression")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")

    args = parser.parse_args()

    app = MLPipelineApp(random_state=args.random_state, cv_folds=args.folds)
    results = app.run(file_path=args.file, target_column=args.target, task_type=args.task)
    print(f"Results:\n{results}")
