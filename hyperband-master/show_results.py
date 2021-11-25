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


for r in sorted(results, key = lambda x: x['lda_score'], reverse=True)[:results_to_show]:
    print( "lda_score: {:.4} | {} seconds | {:.1f} n_doc | run {} ".format( 
                 r['lda_score'], r['seconds'], r['n_doc'], r['counter']))
    pprint(r['params'])
    pprint(r['topic_keywords'])
    print()
