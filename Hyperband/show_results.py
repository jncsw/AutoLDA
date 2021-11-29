#!/usr/bin/env python

"load pickled results, show the best"

import sys
import pickle

from pprint import pprint

def print_topicwords(topicwords):
    
    for (i, t) in enumerate(topicwords):
        print('\nTopic {}:'.format(i+1), end=' ')
        for w in t:
            print(w, end=' ')
    print()

def main():
    try:
    	input_file = sys.argv[1]
    except IndexError:
    	print ("Usage: python show_results.py <results.pkl> [<number of results to show>]\n")
    	raise SystemExit

    try:
    	results_to_show = int( sys.argv[2] )
    except IndexError:
    	results_to_show = 10

    with open( input_file, 'rb' ) as i_f:
    	results = pickle.load( i_f )


    if results_to_show > 0: # from low to high
        print("The best {} configs are:\n".format(results_to_show))
        configs_show = sorted(results, key = lambda x: x['loss'], reverse=False)[:results_to_show]

    else:# from high to low
        print("The worst {} configs are:\n".format(-results_to_show))
        configs_show = sorted(results, key = lambda x: x['loss'], reverse=True)[:-results_to_show]

    for r in configs_show: 
        print( "loss: {:.4} | perplexity_train: {} | {} seconds | {:.1f} n_iter | run {} ".format( 
                     r['loss'], r['perplexity_train'], r['config_seconds'], r['iterations'], r['counter']))
        pprint(r['params'])
        print('topic_words in train_data:')
        print_topicwords(r['topic_keywords'])
        print()    

    # save best 10 configs
    best_10 = sorted(results, key = lambda x: x['loss'], reverse=False)[:10]
    print ("saving best_10 ......")
    with open('best_10_configs.pkl', 'wb') as f:
        pickle.dump(best_10, f)

    # save worst 10 configs
    worst_10 = sorted(results, key = lambda x: x['loss'], reverse=True)[:10]
    print ("saving worst_10 ......")
    with open('worst_10_configs.pkl', 'wb') as f:
        pickle.dump(worst_10, f)

if __name__ == '__main__':
    main()