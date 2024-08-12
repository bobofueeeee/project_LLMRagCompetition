import pandas as pd
from untils.sqlite_toolkit import sqliteToolkit

def sqlwithexcle_excute(sqlwithexcle_path,db_source,sqlwithexcle_excuteresult_path):

    df = pd.read_excel(sqlwithexcle_path)
    print(df)
    df['result'] = ''

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
    df.to_excel(sqlwithexcle_excuteresult_path, index=False, engine='openpyxl')
    return df

if __name__ == '__main__':
    sqlwithexcle_path = "../data/sql_种子.xlsx"
    db_source = r'D:\wk\data\bs_challenge_financial_14b_dataset\dataset\博金杯比赛数据.db'
    sqlwithexcle_excuteresult_path = '../data/sql_seed_result.xlsx'
    sqlwithexcle_excute(sqlwithexcle_path,db_source,sqlwithexcle_excuteresult_path)