import pandas as pd
import mysql.connector as mariadb
from sqlalchemy import create_engine

from langchain_community.utilities import SQLDatabase


class DBManager:
    def __init__(self, conf, host="172.31.40.58", port="23306", user="gi", password="gi"):
        self.conf = conf
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = self.conf.plnt_cd

    def select_from_table(self, query):
        my_cnx = self.__connect()
        my_cursor = my_cnx.cursor()

        try:
            my_cursor.execute(query)
            cols = [i[0] for i in my_cursor.description]
            df = pd.DataFrame(my_cursor.fetchall(), columns=cols)
        except Exception as e:
            print(e)
        finally:
            my_cursor.close()
            my_cnx.close()

        return df

    def __connect(self):
        my_cnx = mariadb.connect(
            host=self.host,
            port=int(self.port),
            user=self.user,
            password=self.password,
            database=self.database,
        )

        return my_cnx

    def get_db_connection(self):
        return SQLDatabase(engine=self.__create_engine())

    def __create_engine(self):
        return create_engine(
            f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )
