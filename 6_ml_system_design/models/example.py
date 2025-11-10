
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import seaborn as sns

# ------------------------
# 1. Configure MLflow
# ------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # your running server
mlflow.set_experiment("Titanic_Experiments")

# ------------------------
# 2. Load and preprocess data
# ------------------------
titanic = sns.load_dataset('titanic')
titanic = titanic.dropna(subset=['age', 'fare', 'embarked', 'sex', 'class', 'survived'])

# Define features
numeric_features = ['age', 'fare']
categorical_features = ['sex', 'class', 'embarked']

# Split
X = titanic[numeric_features + categorical_features]
y = titanic['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
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

# ------------------------
# 3. Define models
# ------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model_name = None
best_f1 = -1

for name, model in models.items():
    print(f"🔹 Running experiment for {name}...")

    with mlflow.start_run(run_name=name):
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba) if proba is not None else None

        # Log parameters and metrics
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        if auc:
            mlflow.log_metric("roc_auc", auc)

        signature = infer_signature(X_test, preds)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name="TitanicModel"
        )

        print(f"✅ {name}: acc={acc:.3f}, f1={f1:.3f}, auc={auc:.3f if auc else None}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline

print(f"\n🏆 Best model: {best_model_name} (F1={best_f1:.3f})")
