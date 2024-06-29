file_name = 'stopwords.txt'

# 初始化一个空列表来保存每一行的内容
stopwords = []

# 使用 'with' 语句来打开文件，这样可以确保文件在使用后被正确关闭
with open(file_name, 'r', encoding='utf-8') as file:
    # 逐行读取文件内容
    for line in file:
        # 去除行尾的换行符（如果有的话）
        stopword = line.strip()
        # 将处理后的行添加到列表中
        stopwords.append(stopword)



