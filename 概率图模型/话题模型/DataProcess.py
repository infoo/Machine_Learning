import jieba
import os
from sklearn.feature_extraction.text import CountVectorizer


class DataProcess(object):
    def __init__(self, infile, outfile=None):
        self._infile = infile
        self._outfile = outfile
        self._stopwords = None

    @property
    def stopwords(self):
        if self._stopwords is None:
            raise ValueError("stopwords is None")
        return self._stopwords

    @stopwords.setter
    def stopwords(self, infile):
        if os.path.exists(infile) is False:
            raise FileNotFoundError(infile + " is not Found")
        self._stopwords = list()
        with open(infile, 'r', encoding='utf8') as fp:
            for word in fp.readlines():
                word = word.strip()
                self._stopwords.append(word)
            fp.close()

    def _docs(self):
        docs = list()
        with open(self._infile, 'r', encoding='utf8') as fp:
            for raw_doc in fp.readlines():
                doc_cut_list = jieba.cut(raw_doc)
                _doc = ' '.join(doc_cut_list)
                docs.append(_doc)
            fp.close()
        return docs

    # 获取词频向量
    def tf(self):
        tf_vec = CountVectorizer(stop_words=self.stopwords)
        tf = tf_vec.fit_transform(self._docs())
        return tf, tf_vec
