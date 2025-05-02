_Based on Python v3.10_

### Installation
```shell
pip install -r requirements.txt
```

### .env.example
> Переименуйте файл .env.exmaple в .env, чтобы заработало
```text
DATABASE_DRIVER=postgresql
DATABASE_USER=postgres
DATABASE_PASSWORD=root
DATABASE_HOST=127.0.0.1
DATABASE_PORT=5432
DATABASE_NAME=cluster

DATABASE_STRING=${DATABASE_DRIVER}://${DATABASE_USER}:${DATABASE_PASSWORD}@${DATABASE_HOST}:${DATABASE_PORT}/${DATABASE_DATABASE}

VK_APP_ID=<id приложения vk api>
VK_ACCESS_TOKEN=<access token из uri при запуске приложения в режиме login>
VK_GROUP_ID=<id группы с которой парсить посты>
VK_POSTS_TO_CHECK=<количество постов спарсить из группы>
```

### cluster.sql
> backup БД PostgreSQL, на который тестировалась программа

### Running
Если через bash-консоль
```shell
python3 -m venv venv
(linux) source venv/bin/activate
(windows) source venv/Scripts/activate
python3 ./main.py
```
Если через Pycharm
```shell
python3 ./main.py
```

### Modes
Запуск авторизации
```shell
python3 ./main.py login
```
Запуск парсинга
```shell
python3 ./main.py parse
```
Запуск кластеризации
```shell
python3 ./main.py cluster
```

### Embeddings Issue
> https://github.com/UKPLab/sentence-transformers/issues/1883
</br>
> https://github.com/UKPLab/sentence-transformers/issues/1356
