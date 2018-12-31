from 话题模型.DataProcess import DataProcess
from sklearn.decomposition import LatentDirichletAllocation


class LDA(object):
    def __init__(self, _tf):
        self._tf = _tf
        self._lda = None
        self._docs_res = None

    @property
    def lda(self):
        if self._lda is None:
            raise ValueError("self._lda is None")
        return self._lda

    @property
    def docs_res(self):
        if self._docs_res is None:
            raise ValueError("self._lda is None")
        return self._docs_res

    def gen_docs_res(self, n_components, learning_offset=50., random_state=0):
        self._lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                              learning_method='online',
                                              learning_offset=learning_offset,
                                              random_state=random_state)
        self._docs_res = self._lda.fit_transform(self._tf)
        return self._docs_res

    def show_result(self, infile):
        docs = list()
        with open(infile, 'r', encoding='utf8') as fp:
            for doc in fp.readlines():
                d = doc.strip()
                docs.append(d)
            fp.close()
        i = 0
        for res in self.docs_res:
            c = res.argsort()[::-1]
            print(str(c[0]) + " : " + str(docs[i]))
            i = i + 1

    def print_top_words(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()


if __name__ == "__main__":
    dataProcess = DataProcess('data/raw_data.txt', 'out_data.txt')
    dataProcess.stopwords = 'data/stopwords.txt'
    tf, tf_vec = dataProcess.tf()

    lda = LDA(tf)
    lda.gen_docs_res(n_components=3)
    lda.print_top_words(lda.lda, tf_vec.get_feature_names(), 10)
    lda.show_result('data/raw_data.txt')
