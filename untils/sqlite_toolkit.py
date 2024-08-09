import sqlite3

class sqliteToolkit:
    def __init__(self,source_path):
        self.conn = sqlite3.connect(source_path)

    def query(self,query):
        cur = self.conn.cursor()
        cur.execute(query)
        result = cur.fetchall()
        return result

if __name__ == '__main__':
    db_source = r'D:\wk\data\bs_challenge_financial_14b_dataset\dataset\博金杯比赛数据.db'
    sqlite_client = sqliteToolkit(db_source)
    query = '''SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;'''
    result = sqlite_client.query(query)
    print(result)
