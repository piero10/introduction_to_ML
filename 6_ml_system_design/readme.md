## Проект пример реализайии ML-модели в отдельном сервисе

### Запускаем сервер


```
cd server
python app_unicorn.py
```

Сервер доступен по адресу:
http://127.0.0.2:8008 

сваггер:
http://127.0.0.2:8008/docs


- Изучаем Swagger
    - http://127.0.0.1:8000/docs
    - видим 2 endpoints
    - схема совпадает с нашей pydantic моделью
    - пример в swagger
    - пример в post
    - выполняем какой-то запрос, видим, что результат поменялся, в зависимости от возраста

- Дергаем сервер из браузера
    - http://127.0.0.1:8000/predict?age=22&fare=7.25&sex=male&pclass=Third&embarked=S
    - http://127.0.0.2:8008/predict_batch?age=29&age=22&fare=35.5&fare=7.25&sex=female&sex=male&pclass=Second&pclass=Third&embarked=S&embarked=C
    - http://127.0.0.2:8008/predict_simple?age=29&fare=35.5&sex=female&pclass=Second&embarked=S
    - разбираемся в запросе

- Реализуем клиент который будет обращаться к серверу
    - для отправки запроса для одного клиента
    - для множества клиентов
    - работаем с клиентом с дебагером

- Запускаем сервер из-под дебагера
    - app_unicorn
        - http://127.0.0.2:8008/
