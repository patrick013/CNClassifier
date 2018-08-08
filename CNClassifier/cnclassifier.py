# READ ME:
# This script aims to classify clinical notes to its corresponding category
# The default categories are Medications, Hospital Course, Laboratories, Physical Examinations and History
"""
classifier_model=classifier()
classifier_model.letspredict(note)
"""
# You could also choose different features for training set
# classifier(datanum, dataset,labels_index,labels_name,algo,feature,tfidf=0)
# datanum : integer, 1-10, the proportion of the dataset used for training classifier model. For instance: datanum=7, 70% dataset would be used in training set
# feature: string, feature="bow", BOW ; feature="skip-gram", Skip-gram; feature="cbow", CBOW;
# tfidf: 0 or 1, when tfidf=1, tfidf would be used while 0 means not use
# dataset: string, the directory of dataset
# labels_index: list, labels or tags for each documents
# labels_nameL list, each label's corresponding category
# algo: string, "mult_nb": multinomial NB, "line_svm": linear SVM
"""
If you want to train your own dataset:
classifier_model=classifier(dataset="Your dataset directory",labels_index=['your','labels'],labels_name=['your','category'])
The format of the dataset document should be:
    label1 This is the first notes of category 1,
    label1 This is the second note of category 1,
    label2 I am the first one in category 2,
    label3 I am in category 3,
"""

import random,re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from gensim.models import Word2Vec
from nltk import word_tokenize
from CNClassifier import helper
import numpy as np

class _loaddata(object):

    def __init__(self,
                 datanum=7,
                 dataset='./dataset/dataset.txt',
                 labels_index=['A','B','C','D','E'],
                 labels_name=["History","Laboratory","Medications","Physical Examinations","Hospital Course"],
                 tfidf=0):
        self.__labels_index=labels_index
        data_set = [line.strip() for line in open(dataset)]
        random.shuffle(data_set)
        num = len(data_set)
        self.__dataset = data_set[:int(datanum * num / 10)]
        self.labels_name=labels_name
        self.tfidf=tfidf

    def __lable2id(self,lable):
        for i in range(len(self.__labels_index)):
            if lable == self.__labels_index[i]:
                return i
        raise Exception('Error lable %s' % (lable))

    def __to_tfidf(self,X,Y):
        transformer = TfidfTransformer(smooth_idf=False)
        tfidf = transformer.fit_transform(X)
        return tfidf.toarray(), Y

    def bow(self):
        vectorizer = CountVectorizer()
        data = [re.split(" +", doc, 1)[1] for doc in self.__dataset]
        labels = [self.__lable2id(re.split(" +", doc, 1)[0]) for doc in self.__dataset]
        X = vectorizer.fit_transform(data)
        if self.tfidf==0:
            return X.toarray(), np.array(labels),vectorizer
        else:
            tfx,tfy=self.__to_tfidf(X.toarray(), np.array(labels))
            return tfx,tfy,vectorizer

    def w2v(self,sg_num=0):
        data = [re.split(" +", doc, 1)[1] for doc in self.__dataset]
        labels = [self.__lable2id(re.split(" +", doc, 1)[0]) for doc in self.__dataset]
        tokenized_documents = [word_tokenize(doc) for doc in data]
        w2v_model = Word2Vec(tokenized_documents, size=100, window=5, min_count=1, sg=sg_num, workers=4)
        X=np.array([np.mean(
                [w2v_model[w] for w in words if w in w2v_model] or [np.zeros(len(w2v_model.itervalues().next()))], axis=0) for
                            words in data])
        if self.tfidf==0:
            return X, np.array(labels), w2v_model
        else:
            tfx, tfy = self.__to_tfidf(X, np.array(labels))
            return tfx,tfy,w2v_model

class classifier(_loaddata):

    def __init__(self,datanum=7,
                 dataset='./dataset/dataset.txt',
                 labels_index=['A','B','C','D','E'],
                 labels_name=["History", "Laboratory", "Medications", "Physical Examinations", "Hospital Course"],
                 algo="mult_nb",
                 feature="bow",
                 tfidf=0):

        _loaddata.__init__(self,datanum,dataset,labels_index,labels_name,tfidf)
        self.__feature=feature
        self.__algo=algo

    def __featureInit(self):
        if self.__feature=="bow":
            return self.bow()
        if self.__feature=="skip-gram":
            return self.w2v(sg_num=1)
        if self.__feature=="cbow":
            return self.w2v( sg_num=0)

    def __trainModel(self):
        x, y ,ftrInitializer= self.__featureInit()
        if self.__algo=="mult_nb":
            cls=MultinomialNB()
            cls.fit(x, y)
        if self.__algo=="line_svm":
            cls=svm.LinearSVC()
            cls.fit(x, y)
        return cls,ftrInitializer

    def __totfidf(self,X):
        transformer = TfidfTransformer(smooth_idf=False)
        tfidf = transformer.fit_transform(X)
        return tfidf.toarray()

    def letspredict(self,Note):
        note= helper.note_cleanser(Note)
        classifier_model,ftrInitializer=self.__trainModel()
        if self.tfidf==0:
            if self.__feature=="bow":
                x = ftrInitializer.transform([note]).toarray()
            if self.__feature=="skip-gram" or self.__feature=="cbow":
                words=word_tokenize(note)
                x=[np.mean(
                    [ftrInitializer[w] for w in words if w in ftrInitializer] or [np.zeros(len(ftrInitializer.itervalues().next()))],
                    axis=0)]
        else:
            if self.__feature=="bow":
                x = self.__totfidf(ftrInitializer.transform([note]).toarray())
            if self.__feature=="skip-gram" or self.__feature=="cbow":
                words=word_tokenize(note)
                x=self.__totfidf([np.mean(
                    [ftrInitializer[w] for w in words if w in ftrInitializer] or [np.zeros(len(ftrInitializer.itervalues().next()))],
                    axis=0)])
        return self.labels_name[classifier_model.predict(x)[0]]
