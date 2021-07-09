import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import os
from mainSystem.model.data_load import item_load
from mainSystem.model.data_pre import save_path


def i2i_sim():

    trainfile = item_load()
    corpus = trainfile['content'].values.astype('U')

    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,  # max_features=n_features,
                                       stop_words='english')

    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        tfidf_vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵

    SimMatrix = (tfidf * tfidf.T).A
    SimMatrix = pd.DataFrame(SimMatrix, index=trainfile.news_id, columns=trainfile.news_id)
    pickle.dump(SimMatrix, open(save_path + r'\emb_i2i_sim.pkl', 'wb'))

    return SimMatrix

