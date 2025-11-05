import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import dagshub

# Initialize DagsHub connection (updated repo name)
dagshub.init(
    repo_owner='malaychand',
    repo_name='daghub-connect',
    mlflow=True
)

# Set MLflow tracking URI for DagsHub
mlflow.set_tracking_uri("https://dagshub.com/malaychand/daghub-connect.mlflow")

# Optional verification
print("MLflow tracking URI set to:", mlflow.get_tracking_uri())

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model parameters
max_depth = 1

# Set up MLflow experiment
mlflow.set_experiment('iris-dt')

with mlflow.start_run():
    # Train the Decision Tree
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    # Predict
    y_pred = dt.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    # Log metrics and parameters
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save and log confusion matrix
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # ✅ Save and log model manually to avoid DagsHub unsupported endpoint error
    joblib.dump(dt, "decision_tree_model.pkl")
    mlflow.log_artifact("decision_tree_model.pkl")

    # Add useful tags
    mlflow.set_tag('author', 'malaychand')
    mlflow.set_tag('model', 'DecisionTreeClassifier')

    # Optional: log script (comment if running interactively)
    # mlflow.log_artifact(__file__)
