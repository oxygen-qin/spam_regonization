1.0 : 基本功能

2.0 : 优化分词和停用词过滤

对tokenization.py中tokenize(text, filtered_words)函数的优化

    token = token.lemma_.lower()
    if str(token).isalpha() and token not in stopwords:
        if token not in filtered_words:
            filtered_words[token] = 1
        else:
            filtered_words[token] = filtered_words[token] + 1
    优化为：
    lemma = token.lemma_.lower()
    if lemma.isalpha() and lemma not in stopwords:
        filtered_words[lemma] = filtered_words.get(lemma, 0) + 1

2.1 : 平滑处理

在计算贝叶斯概率时，可以使用拉普拉斯平滑来避免零概率问题。
    
    # 添加拉普拉斯平滑
    for word, num in spam_word.items():
        p_spam_word[word] = (num + 1) / (spam_num + len(spam_word))
    
    for word, num in ham_word.items():
        p_ham_word[word] = (num + 1) / (ham_num + len(ham_word))

3.0 : 算法优化-使用TF-IDF

词频-逆文档频率（TF-IDF）是一种更先进的文本表示方法，可以提高分类器的性能。可以使用scikit-learn库中的TfidfVectorizer来实现。
但是没有使用前面版本的分词函数（tokenization中的tokenize()）

3.1 : 更新正则表达式

    data['Message'] = data['Message'].str.replace(r'\W', '', regex=True)  # 去除特殊字符
    data['Message'] = data['Message'].str.replace(r'\s+', ' ', regex=True)  #

3.2 : tokenize()重写

