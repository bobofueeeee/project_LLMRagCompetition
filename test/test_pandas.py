import pandas as pd

# 假设df是你的DataFrame
# 创建一个示例DataFrame
data = {'Name': ['Tom', 'Jane', 'Steve', 'Ricky'],
        'Age': [28, 34, 29, 42]}
df = pd.DataFrame(data)

# 新增一列，这里直接赋值一个常数列
df['NewColumn'] = 'Constant Value'

# 或者基于现有列的计算结果新增一列
df['AgePlus10'] = df['Age'] + 10

print(df)
print(type(df))
#------------------------------------------------------------------------

df_result = pd.DataFrame(columns=['question','sql','system_prompt'])
df_result.loc[0] = ['123', '234', '567']
print(df_result)
print(type(df_result))