from gensim import corpora
from gensim import models

import time

class ExpModel(object):

    NUM_TOPICS = 100

    def __init__(self):
        self.dic = corpora.Dictionary()

    def update(self, train_data):
        if not train_data: return

        t0 = time.time()

        other_dic = corpora.Dictionary(train_data)
        self.dic.merge_with(other_dic)
        corpus = [self.dic.doc2bow(text) for text in train_data]

        if not self.__dict__.has_key("lda"):
            self.lda = models.LdaModel(corpus, id2word=self.dic, num_topics=ExpModel.NUM_TOPICS)
        else:
            self.lda.update(corpus)

        #for i in range(ExpModel.NUM_TOPICS): print self.lda.print_topic(i)
        t1 = time.time()
        print 'Training Data Elapsed %fs.' % (t1-t0)
