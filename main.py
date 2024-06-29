import pandas as pd
from tfidf import compute_tf, compute_idf, compute_tfidf
from collections import defaultdict
import math
import joblib

# 读取数据
data = pd.read_csv('spam.csv')

# 数据预处理
data['Message'] = data['Message'].str.lower()  # 转换为小写
data['Message'] = data['Message'].str.replace(r'\W', '', regex=True)  # 去除特殊字符
data['Message'] = data['Message'].str.replace(r'\s+', ' ', regex=True)  # 去除多余的空格

# 分割数据集
# 从原始数据集中随机抽取80%的样本作为训练集，并重置索引（删除旧索引）
train_data = data.sample(frac=0.8, random_state=42).reset_index(drop=True)
# 从原始数据集中删除已经在训练集中出现的索引（即那些被选为训练集的样本），
# 然后重置剩余数据的索引（删除旧索引），这部分数据作为测试集
test_data = data.drop(train_data.index).reset_index(drop=True)

# 计算训练集的TF-IDF
# 将训练数据集中的'Message'列转换为Python列表，每个元素是原始文本消息
train_texts = train_data['Message'].tolist()
# 计算训练文本集合的逆文档频率（IDF）
idfs = compute_idf(train_texts)
train_tfidf = [compute_tfidf(compute_tf(text), idfs) for text in train_texts]

# 检查train_tfidf的长度
# print(f"Length of train_data: {len(train_data)}")
# print(f"Length of train_tfidf: {len(train_tfidf)}")

# 贝叶斯分类器训练
ham_word_counts = defaultdict(int)
spam_word_counts = defaultdict(int)
ham_count = 0
spam_count = 0

for i, row in train_data.iterrows():
    tfidf = train_tfidf[i]
    if row['Category'] == 'ham':
        ham_count += 1
        for word, value in tfidf.items():
            ham_word_counts[word] += value
    else:
        spam_count += 1
        for word, value in tfidf.items():
            spam_word_counts[word] += value

# 计算先验概率
# 计算每个类别的先验概率。
p_ham = ham_count / len(train_data)
p_spam = spam_count / len(train_data)


# 计算条件概率
# 计算每个词在给定类别下的条件概率，使用拉普拉斯平滑来避免零概率问题。
def compute_prob(word_counts, total_count, vocab_size):
    prob_dict = defaultdict(float)
    for word, count in word_counts.items():
        prob_dict[word] = (count + 1) / (total_count + vocab_size)
    return prob_dict


vocab_size = len(set(ham_word_counts.keys()).union(set(spam_word_counts.keys())))
ham_probs = compute_prob(ham_word_counts, ham_count, vocab_size)
spam_probs = compute_prob(spam_word_counts, spam_count, vocab_size)

# 保存模型参数
joblib.dump(idfs, 'idfs.pkl')
joblib.dump(ham_probs, 'ham_probs.pkl')
joblib.dump(spam_probs, 'spam_probs.pkl')
joblib.dump(p_ham, 'p_ham.pkl')
joblib.dump(p_spam, 'p_spam.pkl')
joblib.dump(ham_count, 'ham_count.pkl')
joblib.dump(spam_count, 'spam_count.pkl')
joblib.dump(vocab_size, 'vocab_size.pkl')


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


# 测试模型
test_texts = test_data['Message'].tolist()
test_labels = test_data['Category'].tolist()
predictions = [predict(text) for text in test_texts]

# 计算准确率
accuracy = sum(1 for i in range(len(predictions)) if predictions[i] == test_labels[i]) / len(predictions)
print("Accuracy:", accuracy)