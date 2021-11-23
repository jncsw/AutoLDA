#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @time : 11/17/2021 10:20 AM
# @Author: DONGQING YANG
# @File : .py

import numpy as np
# from gensim.models import LdaModel
from GenerateEmbeddings import GenEmb, Calc_Dist


def embedding_distance(topic_words, model):

    allModel = ["BERT", "GLOVE", "W2V", "ELMo"]
    for model in allModel:
        embeddings = GenEmb(topic_words, model)
        print("------ Embedding for ", model, "------", len(embeddings))
        # print("Calculating distance...")
        # print("Score = ", Calc_Dist(AllEmb))
        #score = Calc_Dist(AllEmb)



    # distance
    score = cal_distance(embeddings, 'coh-dif')

    return score


def cal_distance(embeddings, method):

    score = 0

    return score


def jaccard(self,list1,list2):
    intersection=len(list(set(list1).intersection(list2)))
    union=(len(list1)+len(list2))-intersection
    return float(intersection)/union

def get_stability(self,topic_matrix):
    mean=[]
    row=topic_matrix.shape[0]
    for i in range(row):
        mean.append(np.mean(topic_matrix[i]))
    sum=0
    for i in range(row):
        sim=jaccard(row[i],mean)
        sum+=sim
    stability=sum/np.sum(topic_matrix)
    
    return stability


def coefficient_of_variation(self,list):
    mean=np.mean(list)
    std=np.std(list,ddof=0)
    cv=std/mean
    return cv

def get_variability(self,topic_matrix):
    row=topic_matrix.shape[0]
    cv_list=[]
    for i in range(row):
        cv=coefficient_of_variation(topic_matrix[i])
        cv_list.append(cv)
    variability=np.std(cv_list,ddof=0)
    return variability


def log_perplexity(self, chunk, total_docs=None):
    if total_docs is None:
        total_docs=len(chunk)
    corpus_words=sum(cnt for document in chunk for _,cnt in document)
    subsample_ratio=1.0*total_docs/len(chunk)
    perwordbound=self.bound(chunk,subsample_ratio=subsample_ratio)/(subsample_ratio * corpus_words)
    perplexity=np.exp2(-perwordbound)
    
    return perplexity

#calulate the perplexity of LDA model
# lda=LdaModel(common_corpus,num_topics=num_topic,id2word=idc,alpha='auto',chunksize=len(texts_all),iterations=20000)
# perplexity=lda.log_perplexity(common_corpus)
