"""
分词处理
"""
# 导入spacy包用于分词
import spacy
# 读入停用词表
from stopwords import stopwords

# 加载预训练的英语Spacy模型
"""
'en_core_web_sm'：这是一个预训练的英语模型。\
这里的'en'代表英语，' core '代表这是一个核心模型，
' web '代表它包含一些针对网页文本的特定功能，
而'sm'代表它是一个小模型（size medium的缩写，
但实际上它是最小的预训练模型
"""
nlp = spacy.load('en_core_web_sm')


# 测试分词的数据
# example = ("Go until jurong point, crazy.. Available only in bugis "
#            "n great world la e buffet... Cine there got amore wat...")
# filtered_words = {}
def tokenize(text, filtered_words):
    """

    :param text: 英文文本信息
    :param filtered_words: 存储分词结果的dict
    :return: 没有返回
    """
    doc = nlp(text)
    for token in doc:
        # token = token.lemma_.lower()
        # if str(token).isalpha() and token not in stopwords:
        #     if token not in filtered_words:
        #         filtered_words[token] = 1
        #     else:
        #         filtered_words[token] = filtered_words[token] + 1
        lemma = token.lemma_.lower()
        if lemma.isalpha() and lemma not in stopwords:
            filtered_words[lemma] = filtered_words.get(lemma, 0) + 1

# tokenize(example, filtered_words)
# print(filtered_words)
