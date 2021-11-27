#!/usr/bin/env python

"load pickled results, show the best"

import sys
import pickle

from pprint import pprint

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


# from low to high
if results_to_show > 0:
    print("The best {} configs are:\n".format(results_to_show))
    for r in sorted(results, key = lambda x: x['loss'], reverse=False)[:results_to_show]: # from low to high
        print( "loss: {:.4} | perplexity_train: {} | {} seconds | {:.1f} n_iter | run {} ".format( 
                     r['loss'], r['perplexity_train'], r['config_seconds'], r['iterations'], r['counter']))
        pprint(r['params'])
        pprint(r['topic_keywords']) # in training
        print()   

else:
    print("The worst {} configs are:\n".format(-results_to_show))
    for r in sorted(results, key = lambda x: x['loss'], reverse=True)[:-results_to_show]: # from low to high
        print( "loss: {:.4} | perplexity_train: {} | {} seconds | {:.1f} n_iter | run {} ".format( 
                     r['loss'], r['perplexity_train'], r['config_seconds'], r['iterations'], r['counter']))
        pprint(r['params'])
        pprint(r['topic_keywords'])
        print()    
