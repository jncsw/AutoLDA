import pickle
from lda import print_params, LDA_classifier
from show_results import print_topicwords
import sys

def run_config(n_iterations, params, data):

    print('\n'+'--'*10)
    print_params(params)
    
    # run LDA on data
    lda = LDA_classifier(n_iterations, params)
    
    t_w, n_iter, perplexity = lda.fit(data)
    
    # lda_score = lda.semantic_score(t_w, emb_model)
    # perplexity = lda.score(data) 

    print('perplexity = {}'.format(perplexity))
    print_topicwords(t_w)

    return {'params': params,'perplexity': perplexity, 'topic_keywords': t_w, 'n_iter':n_iter}

if __name__ == '__main__':
    with open( 'best_10_configs.pkl', 'rb' ) as f:
        best_10 = pickle.load( f )

    with open( 'worst_10_configs.pkl', 'rb' ) as f:
        worst_10 = pickle.load( f )

    print('loading full data')
    with open('data.pkl','rb') as pk:
        data = pickle.load(pk)


    num = int(sys.argv[1])
    print('num =',num)

    print('\n'+'==='*10+' best {}:'.format(num))
    good_res = []
    for idx,good in enumerate(best_10):
    
        if idx == num:
            break
        print('idx =',idx)
        res = run_config(20, good['params'], data)
        good_res.append(res)


    print('\n'+'==='*10+' worst {}:'.format(num))
    bad_res = []
    for idx, bad in enumerate(worst_10):

        if idx == num:
            break
        print('idx =',idx)
        res = run_config(20, bad['params'], data)
        bad_res.append(res)

    print ("saving best res ......")
    with open('best_{}_configs_res.pkl'.format(num), 'wb') as f:
        pickle.dump(good_res, f)

    print ("saving worst res ......")
    with open('worst_{}_configs_res.pkl'.format(num), 'wb') as f:
        pickle.dump(bad_res, f)