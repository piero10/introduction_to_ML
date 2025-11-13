import pandas as pd
import requests
from urllib.parse import urlencode

# Пример данных: несколько пассажиров
data = pd.DataFrame([
    {"age": 22, "fare": 7.25, "sex": "male", "pclass": "Third", "embarked": "S"},
    {"age": 38, "fare": 71.2833, "sex": "female", "pclass": "First", "embarked": "C"},
    {"age": 26, "fare": 7.925, "sex": "female", "pclass": "Third", "embarked": "S"},
])

# Базовый URL вашего FastAPI сервера
base_url = "http://127.0.0.1:8000/predict_batch?"

# 🔹 Формируем параметры GET-запроса
params = []
for col in data.columns:
    for val in data[col]:
        params.append((col, val))

# Кодируем параметры в URL
query_string = urlencode(params)
url = base_url + query_string

print("GET URL:", url)

# Отправляем GET-запрос
response = requests.get(url)
if response.status_code == 200:
    predictions = response.json()
    # Добавляем результаты в DataFrame
    data["survived"] = [p["survived"] for p in predictions]
    data["probability_of_survival"] = [p["probability_of_survival"] for p in predictions]
    print(data)
else:
    print("Ошибка:", response.text)
