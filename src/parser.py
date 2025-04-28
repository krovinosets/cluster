import math
import os

import requests
import psycopg2
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()


class Parser:
    def __init__(self, token: str):
        self.token = token
        self.base_url = 'https://api.vk.com/method/'

        self.db_params = {
            'dbname': os.getenv('DATABASE_NAME'),
            'user': os.getenv('DATABASE_USER'),
            'password': os.getenv('DATABASE_PASSWORD'),
            'host': os.getenv('DATABASE_HOST'),
            'port': os.getenv('DATABASE_PORT')
        }

    def get_posts(self, owner_id: str, count: int) -> List[Dict] | None:
        """Получение постов из паблика"""

        print(count)
        posts: list[dict] = []
        print(math.ceil(count / 100))
        for cycle in range(0, math.ceil(count / 100)):
            print(cycle)
            method = 'wall.get'
            url = f"{self.base_url}{method}?owner_id=-{owner_id}&offset={cycle*100}&count={count}&access_token={self.token}&v=5.131"

            try:
                response = requests.get(url)
                data = response.json()
                posts.extend(data['response']['items'])
                print(len(posts))
            except Exception as e:
                print(f"Ошибка при получении данных: {e}")
                return None
        return posts

    def save_to_db(self, posts: List[Dict]):
        """Сохранение постов в PostgreSQL"""
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()

        try:
            for post in posts:
                if post['text'] == "":
                    continue
                cursor.execute("""
                    INSERT INTO vk_posts 
                    (post_id, text, date, likes, reposts, views)
                    VALUES (%s, %s, to_timestamp(%s), %s, %s, %s)
                    ON CONFLICT (post_id) DO NOTHING
                """, (
                    post['id'],
                    post['text'],
                    post['date'],
                    post['likes']['count'],
                    post['reposts']['count'],
                    post['views']['count']
                ))
            conn.commit()
        except Exception as e:
            print(f"Ошибка записи в БД: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
