# tfidf.py

import math
from collections import defaultdict


def compute_tf(text):
    tf_dict = defaultdict(int)
    words = text.split()
    for word in words:
        tf_dict[word] += 1
    total_words = len(words)
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / total_words  # 这里保持浮点数计算
    return tf_dict


def compute_idf(documents):
    idf_dict = defaultdict(int)
    length = len(documents)
    for doc in documents:
        words = set(doc.split())
        for word in words:
            idf_dict[word] += 1
    for word in idf_dict:
        idf_dict[word] = math.log(length / (idf_dict[word] + 1))  # 这里保持浮点数计算
    return idf_dict


def compute_tfidf(tf, idf):
    tfidf = {}
    for word, tf_val in tf.items():
        tfidf[word] = tf_val * idf.get(word, 0)  # 这里保持浮点数计算
    return tfidf
