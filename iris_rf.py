import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

# Initialize DagsHub connection (updated repo name)
dagshub.init(
    repo_owner='malaychand',
    repo_name='daghub-connect',
    mlflow=True
)

# Set MLflow tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/malaychand/daghub-connect.mlflow")

# Optional: verify the tracking URI
print("MLflow tracking URI set to:", mlflow.get_tracking_uri())

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Random Forest parameters
max_depth = 1
n_estimators = 100

# Create (or get) MLflow experiment
mlflow.set_experiment('iris-rf')

with mlflow.start_run():
    # Initialize and train the Random Forest model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    # Log parameters and metrics to MLflow
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_metric('accuracy', accuracy)

    # Create and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save the plot and log as artifact
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log the trained model
    mlflow.sklearn.log_model(rf, "random-forest-model")

    # Add useful tags
    mlflow.set_tag('author', 'malaychand')
    mlflow.set_tag('model', 'RandomForestClassifier')

    # (Optional) Log this script file itself if running from .py file
    # mlflow.log_artifact(__file__)

print("Run completed and logged successfully.")
