## Пример проекта с mlflow



### просмотр экспериментов:
все делаем в окружении mlflow

```commandline
mlflow ui
```

### Сервис / Serving модели


поднять обученную модель как сервис
```commandline
mlflow models serve -m "runs:/df3eae37fb8c4f01bc227ac9fbc2d4cd/model_rf" -p 8000 --no-conda
```

curl -X POST -H "Content-Type: application/json" -d "{\"dataframe_split\": {\"columns\": [\"Pclass\",\"Sex\",\"Age\",\"Fare\",\"Embarked\"], \"data\": [[3,1,22,7.25,0]]}}" http://127.0.0.1:8000/invocations


