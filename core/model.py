from gensim import models
from gensim import corpora
from gensim import similarities

import time
import utils
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

class ExpModel(object):

    LDA_NUM_TOPICS = 50
    LDA_PASSES = 10
    W2V_NUM_CONTEXT_WORD = 100
    W2V_ITER = 10
    PARA_EXPAND_FACTOR = 1.9

    def __init__(self):
        return

    def Update(self, train_data):
        if not train_data: return

        print 'Start Training Model with %d data.' % len(train_data)
        t0 = time.time()
        try:
            if not self.__dict__.has_key('lda'):
                self.dic = corpora.Dictionary(train_data)
                corpus = [self.dic.doc2bow(text) for text in train_data]
                self.lda = models.LdaMulticore(corpus=corpus, num_topics=ExpModel.LDA_NUM_TOPICS, id2word=self.dic, passes=ExpModel.LDA_PASSES)
                #self.lda = models.LdaModel(corpus, id2word=self.dic, num_topics=ExpModel.LDA_NUM_TOPICS, passes=ExpModel.LDA_PASSES)
                self.index = similarities.Similarity(None, self.lda[corpus], ExpModel.LDA_NUM_TOPICS*2)
            else:
                other_corpus = [self.dic.doc2bow(text) for text in train_data]
                self.lda.update(other_corpus, passes=ExpModel.LDA_PASSES)
                self.index.add_documents(self.lda[other_corpus])
            #for i in range(ExpModel.LDA_NUM_TOPICS): print self.lda.print_topic(i)
            t1 = time.time()
            print 'Training LDA Model Elapsed %dmin %ds.' % utils.float_to_time(t1-t0)

            if not self.__dict__.has_key('w2v'):
                self.w2v = models.Word2Vec(train_data, size=500, window=20, min_count=3, sg=1, iter=ExpModel.W2V_ITER)
            else:
                self.w2v.build_vocab(train_data, update=True)
                self.w2v.train(train_data)
            t2 = time.time()
            print 'Training W2V Model Elapsed %dmin %ds.' % utils.float_to_time(t2-t1)
        except Exception as e:
            print 'Training Model Error: %s' % e.message
            pass

    def Expand(self, keywords):
        probs, context = [], []
        #print "### LEN ###", len(self.w2v.vocab)
        for keyword in keywords:
            if keyword not in self.w2v.vocab:
                print 'Keyword %s not in vocabulary.' % keyword
                continue
            probs.append((keyword, 2.0))
            probs.extend(self.w2v.similar_by_word(keyword, ExpModel.W2V_NUM_CONTEXT_WORD))
        for i,j in probs: print i,j

        if probs:
            min_prob = min(probs, key=lambda x: x[1])[1]
            for p in probs:
                context.extend([p[0]]*int(ExpModel.PARA_EXPAND_FACTOR* p[1] / min_prob))

        return context

    def Seek(self, context, topn=20):
        vec_bow = self.dic.doc2bow(context)
        vec_lda = self.lda[vec_bow]
        sims = self.index[vec_lda]
        sims = sorted(enumerate(sims), key=lambda x: -x[1])
        print sims[:topn]

        return sims[:topn]
