import pandas as pd
import requests

# Пример данных: несколько пассажиров
data = pd.DataFrame([
    {"age": 22, "fare": 7.25, "sex": "male", "pclass": "Third", "embarked": "S"},
    {"age": 38, "fare": 71.2833, "sex": "female", "pclass": "First", "embarked": "C"},
    {"age": 26, "fare": 7.925, "sex": "female", "pclass": "Third", "embarked": "S"},
])

# URL вашего FastAPI сервера
url = "http://127.0.0.1:8000/predict"

# Отправляем прогноз по каждому пассажиру
predictions = []
for _, row in data.iterrows():
    payload = row.to_dict()
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        predictions.append(response.json())
    else:
        predictions.append({"error": response.text})

# Объединяем результаты с исходными данными
results = data.copy()
results["survived"] = [p.get("survived") for p in predictions]
results["probability_of_survival"] = [p.get("probability_of_survival") for p in predictions]

print(results)
