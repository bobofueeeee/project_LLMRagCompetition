test = {"id":1}
print(test['id'])
test['id2']=2
print(test['id2'])

list1 = [1,2]
print(type(list1))

import openpyxl

# 打开工作簿
workbook = openpyxl.load_workbook(r'D:\Users\Desktop\财务简易记账.xlsx')

# 选择工作表（默认情况下，第一个工作表是 active 的）
sheet = workbook.active
cell_value = sheet['A1'].value
print(f"单元格 A1 的值是: {cell_value}")