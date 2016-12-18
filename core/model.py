from gensim import models
from gensim import corpora
from multiprocessing import cpu_count

import time
import utils
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

DIC_MODEL_FILE = 'models/dic.mdl'
TFIDF_MODEL_FILE = 'models/tfidf.mdl'
LDA_MODEL_FILE = 'models/lda.mdl'
W2V_MODEL_FILE = 'models/w2v.mdl'

class ExpModel(object):

    TFIDF_PICK_THRESHOLD = 0.05
    LDA_PICK_THRESHOLD = 0.0005
    LDA_NUM_TOPICS = 50
    LDA_NUM_WORDS = 100
    LDA_PASSES = 10
    W2V_NUM_CONTEXT_WORD = 20
    W2V_ITER = 10

    def __init__(self, load_models=False):
        self.index = {}
        if load_models:
            self.dic = corpora.Dictionary.load(DIC_MODEL_FILE)
            self.tfidf = models.TfidfModel.load(TFIDF_MODEL_FILE)
            self.lda = models.LdaMulticore.load(LDA_MODEL_FILE)
            self.w2v = models.Word2Vec.load(W2V_MODEL_FILE)
            self.getIndexByLDA()
            print 'Loaded Models from Files.'
            self.established = True
        else:
            self.established = False

    def getIndexByLDA(self):
        self.index.clear()
        for i in range(ExpModel.LDA_NUM_TOPICS):
            for i, j in self.lda.get_topic_terms(i, ExpModel.LDA_NUM_WORDS):
                if i not in self.index: self.index[i] = 0.0
                self.index[i] += j

    def validateTokenWithTFIDF(self, token, vec):
        if not vec: return False

        l, r = 0, len(vec)-1
        while l <= r:
            mid = (l+r)>>1
            if vec[mid][0] <= token: l=mid+1
            else: r=mid-1
        return vec[r][0] == token and vec[r][1] >= ExpModel.TFIDF_PICK_THRESHOLD

    def Update(self, train_data):
        if not train_data: return

        self.established = False
        print 'Start Training Model with %d Data.' % len(train_data)
        t0 = time.time()
        try:
            if not self.__dict__.has_key('lda'):
                self.dic = corpora.Dictionary(train_data)
                corpus = [self.dic.doc2bow(text) for text in train_data]
                self.tfidf = models.TfidfModel(corpus=corpus, id2word=self.dic)
                self.lda = models.LdaMulticore(corpus=corpus, num_topics=ExpModel.LDA_NUM_TOPICS, id2word=self.dic, passes=ExpModel.LDA_PASSES)
                #self.lda = models.LdaModel(corpus, id2word=self.dic, num_topics=ExpModel.LDA_NUM_TOPICS, passes=ExpModel.LDA_PASSES)
                #self.index = similarities.Similarity(None, self.lda[corpus], ExpModel.LDA_NUM_TOPICS*2)
            else:
                other_corpus = [self.dic.doc2bow(text) for text in train_data]
                self.lda.update(other_corpus)
                #self.lda.update(other_corpus, passes=ExpModel.LDA_PASSES)
                #self.index.add_documents(self.lda[other_corpus])
            self.getIndexByLDA()

            t1 = time.time()
            print 'Training LDA Model Elapsed %dmin %ds.' % utils.float_to_time(t1-t0)

            if not self.__dict__.has_key('w2v'):
                self.w2v = models.Word2Vec(train_data, size=500, window=10, min_count=3, workers=cpu_count(), sg=1, iter=ExpModel.W2V_ITER)
            else:
                self.w2v.build_vocab(train_data, update=True)
                self.w2v.train(train_data)
            t2 = time.time()
            print 'Training W2V Model Elapsed %dmin %ds.' % utils.float_to_time(t2-t1)
            # print "### LEN ###", len(self.dic), len(self.index), len(self.w2v.vocab)

            self.dic.save(DIC_MODEL_FILE)
            self.tfidf.save(TFIDF_MODEL_FILE)
            self.lda.save(LDA_MODEL_FILE)
            self.w2v.save(W2V_MODEL_FILE)
            t3 = time.time()
            print 'Saving Models Elapsed %dmin %ds.' % utils.float_to_time(t3-t2)

            self.established = True
        except Exception as e:
            print 'Training Model Error: %s' % e.message
            pass

    def Pick(self, tokens):
        if not self.established: return False, None

        vec_bow = self.dic.doc2bow(tokens)
        vec_tfidf = self.tfidf[vec_bow]
        # print "### CHECK ###", min([x[1] for x in vec_tfidf]), max([x[1] for x in vec_tfidf]), min(self.index.values()), max(self.index.values())
        new_tokens = []
        cnt = 0
        for token in tokens:
            token_id = self.dic.token2id[token]
            if token_id not in self.dic:
                print 'Token %s not in vocabulary.' % token
                continue
            if token_id in self.index and self.index[token_id] >= ExpModel.LDA_PICK_THRESHOLD or \
                    self.validateTokenWithTFIDF(token_id, vec_tfidf):
                if token_id in self.index and self.index[token_id] >= ExpModel.LDA_PICK_THRESHOLD: cnt += 1
                new_tokens.append(token)

        # print "### COMPRESS ###", len(tokens), len(new_tokens), cnt, 1.0*len(new_tokens)/len(tokens)
        return True, new_tokens

    def Expand(self, keywords):
        if not self.established: return False, None

        probs = []
        for keyword in keywords:
            if keyword not in self.w2v.vocab:
                print 'Keyword %s not in vocabulary.' % keyword
                continue
            probs.append((keyword, 1.0))
            probs.extend(self.w2v.similar_by_word(keyword, ExpModel.W2V_NUM_CONTEXT_WORD))
        for i,j in probs: print i,j

        if probs: probs.sort(key=lambda x: -x[1])

        return True, probs
