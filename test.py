import pandas as pd
import re

# 读取数据
data = pd.read_csv('spam.csv')

print("-------------")
print(data['Message'])

# 数据预处理
data['Message'] = data['Message'].str.lower()  # 转换为小写

print("-------------")
print(data['Message'])

data['Message'] = data['Message'].str.replace(r'\W', ' ', regex=True)  # 去除特殊字符

print("-------------")
print(data['Message'])

data['Message'] = data['Message'].str.replace(r'\s+', ' ', regex=True)  # 去除多余的空格

print("-------------")
print(data['Message'])