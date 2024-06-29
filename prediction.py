import pandas as pd
from tfidf import compute_tf, compute_tfidf
from collections import defaultdict
import math
import joblib

# 加载训练好的模型参数
idfs = joblib.load('idfs.pkl')
ham_probs = joblib.load('ham_probs.pkl')
spam_probs = joblib.load('spam_probs.pkl')
p_ham = joblib.load('p_ham.pkl')
p_spam = joblib.load('p_spam.pkl')
ham_count = joblib.load('ham_count.pkl')
spam_count = joblib.load('spam_count.pkl')
vocab_size = joblib.load('vocab_size.pkl')


# 预测函数
def predict(text):
    tfidf = compute_tfidf(compute_tf(text), idfs)
    p_ham_given_text = math.log(p_ham)
    p_spam_given_text = math.log(p_spam)

    for word, value in tfidf.items():
        p_ham_given_text += math.log(ham_probs.get(word, 1 / (ham_count + vocab_size))) * value
        p_spam_given_text += math.log(spam_probs.get(word, 1 / (spam_count + vocab_size))) * value

    if p_spam_given_text > p_ham_given_text:
        return 'spam'
    else:
        return 'ham'


# 测试函数
def test():
    data = pd.read_csv('test.csv')
    success = 0
    all = 0
    for index in data.index:
        all += 1
        if predict(data["Message"][index]) == data["Category"][index]:
            success += 1

    print(float(success / all))


test()