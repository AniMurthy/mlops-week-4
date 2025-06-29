import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os

DATA_CSV_PATH = 'data/iris.csv'
ARTIFACTS_DIR = 'artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.joblib')
METRICS_PATH = os.path.join(ARTIFACTS_DIR, 'metrics.txt')

def train_model():
    """
    Loads iris data, trains a Decision Tree, and saves the model
    and metrics locally.
    """
    #Load Data
    data = pd.read_csv(DATA_CSV_PATH)
    print(f"Data loaded successfully. Shape: {data.shape}")
   
    #Define Features and Target
    X_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    y_target = 'species'

    #Split Data
    X = data[X_features]
    y = data[y_target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    #Train the Model
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    print(f"Model trained: {model}")

    #Evaluate
    prediction = model.predict(X_test)
    accuracy = metrics.accuracy_score(prediction, y_test)
    print(f"Model Test Accuracy: {accuracy:.4f}")

    #Save Artifacts
    joblib.dump(model, MODEL_PATH)
    print(f"Trained model saved to: {MODEL_PATH}")

    with open(METRICS_PATH, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
    print(f"Metrics saved to: {METRICS_PATH}")

if __name__ == "__main__":
    print("--- Running Iris Decision Tree Model Training ---")
    train_model()
    print("--- Model Training and Local Saving Complete ---")
