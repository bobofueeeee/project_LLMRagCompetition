import sqlite3

conn = sqlite3.connect(r'D:\wk\data\bs_challenge_financial_14b_dataset\dataset\博金杯比赛数据.db')
cur = conn.cursor()
cur.execute('''SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;''')
tables = cur.fetchall()

print(tables)