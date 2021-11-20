"function (and parameter space) definitions for hyperband"
"LDA"

from common_defs import *
from hyperopt.pyll.stochastic import sample
from load_data_lda import train_data
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.base import BaseEstimator, ClassifierMixin
from random import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# define search space
space = {'max_df': hp.uniform('maxdf', 0.7, 1),
         'min_df': hp.uniform('mindf', 0.1, 0.2),
         'topic_number': 5+hp.randint('tn',15),
         # 'learning_method': hp.choice('lm', ('online', 'batch')),
         'learning_decay': hp.uniform('kappa', 0.51, 1.0),
         'learning_offset': 1 + hp.randint('tau_0', 20),
         'batch_size': hp.choice( 'bs', ( 16, 32, 64, 128, 256 )),
         'max_iter': hp.choice('max_iter',(5, 10, 20))
    }

def get_params():
    params = sample(space)
    return handle_integers(params)

def print_params(params):
    pprint({ k : v for k, v in params.items()})
    return None

def try_params(n_doc, params):
    
    print ("n_doc:", n_doc)
    print_params(params)
    
    # print('train_data =')
    # print(train_data)
    # print(len(train_data))
    
    # select n_doc as resources
    data = train_data[:n_doc] 
    
    # run LDA on data
    lda = LDA_classifier(params)
    t_w = lda.fit(data)
    
    lda_score = lda.semantic_score(t_w)
    print('lda_score = {}'.format(lda_score))

    return {'lda_score': lda_score,'topic_keywords': t_w}


# def show_topics(lda, n_words=10):
#     vectorizer, lda_model = best_LDA.get_model()
    
#     keywords = np.array(vectorizer.get_feature_names())
    
#     topic_keywords = []
#     
#     for topic_weights in lda_model.components_:
#         top_keyword_locs = (-topic_weights).argsort()[:n_words]
#         topic_keywords.append(keywords.take(top_keyword_locs))
#     
#     for i in range(0, len(topic_keywords)):
#         print("Topic " + str(i))
#         print(list(topic_keywords[i]))
#     
#     return topic_keywords

class LDA_classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, params):
        self.max_df = params['max_df']
        self.min_df = params['min_df']
        self.topic_number = params['topic_number']
        self.learning_decay = params['learning_decay']
        self.learning_offset = params['learning_offset']
        self.batch_size = params['batch_size']
        self.max_iter = params['max_iter']
        
    def fit(self, train_data):
        # print('fitting:', self.max_df, self.min_df, self.topic_number)
        self.vectorizer = CountVectorizer(max_df=self.max_df, min_df=self.min_df)
        
        self.lda_model = LatentDirichletAllocation(n_components=self.topic_number, 
                                                   learning_method='online', 
                                                   learning_decay = self.learning_decay,
                                                   learning_offset = self.learning_offset,
                                                   batch_size = self.batch_size,                                                 
                                                   max_iter=self.max_iter,
                                                   random_state=100)
        
        data_vectorized = self.vectorizer.fit_transform(train_data)
        self.lda_model.fit(data_vectorized)
        
        # return the topic_keywords
        
        keywords = np.array(self.vectorizer.get_feature_names())
    
        topic_keywords = []
        n_words = 10
        
        for topic_weights in self.lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        
        for i in range(0, len(topic_keywords)):
            print("Topic " + str(i))
            print(list(topic_keywords[i]))
        
        return topic_keywords

    def predict(self, texts):
        text_vectorized = self.vectorizer.transform(texts)
        return self.lda_model.transform(text_vectorized)

    def score(self, train_data):
        # score = self.lda_model.perplexity(self.data_vectorized)
        tmp = self.vectorizer.transform(train_data)
        score = self.lda_model.perplexity(tmp)
        print('scoring:', score)
        return score
    
    def semantic_score(self, topic_keywords):
        
        score = 0
        #score = glove_score(topic_words)
        score = random()
        
        return score

    def get_model(self):
        # print('the final model:', self.max_df, self.min_df, self.topic_number)
        return (self.vectorizer, self.lda_model)