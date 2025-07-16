import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import psycopg2 as pg2

class DatabaseConnector:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = None
        self.cur = None
        self.user = os.getenv('POSTGRES_USER')
        self.password = os.getenv('POSTGRES_PASSWORD')
        self.connect()

    def connect(self):
        self.conn = pg2.connect(host='localhost', database=self.db_name, user=self.user, password=self.password)
        self.cur = self.conn.cursor()

    def fetch_all_data(self):
        self.cur.execute("SELECT * FROM advanced_housing")
        full_db_data = self.cur.fetchall()
        return pd.DataFrame(full_db_data, columns=[desc[0] for desc in self.cur.description])

if __name__ == 'main':
    dbc = DatabaseConnector(db_name='advanced_housing')
    data = dbc.fetch_all_data()
    a=1

