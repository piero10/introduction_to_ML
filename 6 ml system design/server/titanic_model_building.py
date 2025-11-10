import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# -------------------------------
# 1. Load dataset
# -------------------------------
titanic = sns.load_dataset('titanic').dropna(subset=['age', 'fare', 'embarked', 'sex', 'class', 'survived'])

X = titanic[['age', 'fare', 'sex', 'class', 'embarked']]
y = titanic['survived']

# -------------------------------
# 2. Preprocessing
# -------------------------------
numeric_features = ['age', 'fare']
categorical_features = ['sex', 'class', 'embarked']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# -------------------------------
# 3. Train model
# -------------------------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained successfully!")

# -------------------------------
# 4. Save model
# -------------------------------
joblib.dump(model, "titanic_model.pkl")
print("💾 Model saved to titanic_model.pkl")
