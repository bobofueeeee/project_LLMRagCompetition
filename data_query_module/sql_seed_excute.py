import pandas as pd
from untils.sqlite_toolkit import sqliteToolkit

df = pd.read_excel("../data/sql_种子.xlsx")
print(df)
df['result'] = ''

db_source = r'D:\wk\data\bs_challenge_financial_14b_dataset\dataset\博金杯比赛数据.db'
sqllite_clinet = sqliteToolkit(db_source)

for i in range(len(df)):
    print(f'----------------处理第{i}行数据-----------------')
    query = df.loc[i,'sql']
    print(query)
    try:
        result = sqllite_clinet.query(query)
    except Exception as e:
        print(e)
        result = e
    print(result)
    df['result'][i] = result

print(df)
df.to_excel('../data/sql_seed_result.xlsx', index=False, engine='openpyxl')

