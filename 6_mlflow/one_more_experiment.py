import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import mlflow.sklearn


# Загрузка данных Titanic
df = pd.read_csv("data/train.csv")

# Обработка
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow эксперимент
mlflow.set_experiment("one_more_experiment")

with mlflow.start_run() as run:

    mod = RandomForestClassifier(random_state=42)

    mod.fit(X_train, y_train)
    best_model = mod

    # Метрики на тесте
    preds = best_model.predict(X_test)
    probas = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probas)

    # Логирование в MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", auc)

    # ✅ Логирование модели **внутри активного run**
    input_example = pd.DataFrame([{
        "Pclass": 3,
        "Sex": 1,
        "Age": 22,
        "Fare": 7.25,
        "Embarked": 0
    }])
    mlflow.sklearn.log_model(
        sk_model=mod,
        artifact_path="one_more_model",
        input_example=input_example
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
