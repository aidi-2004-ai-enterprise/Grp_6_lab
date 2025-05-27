import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
 
# Load dataset (can later replace with penguins or custom)
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
 
# Create XGBoost model
model = xgb.XGBClassifier()
 
print(" XGBoost model created successfully!")
