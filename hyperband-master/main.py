#!/usr/bin/env python

"a more polished example of using hyperband"
"includes displaying best results and saving to a file"

import sys
#import cPickle as pickle
import pickle
from pprint import pprint
import numpy as np
# import matplotlib.pyplot as plt

from hyperband import Hyperband, Hyperband_LDA
from Embeddings.GLOVE import load_GLOVE

#from defs.gb import get_params, try_params
#from defs.rf import get_params, try_params
#from defs.xt import get_params, try_params
#from defs.rf_xt import get_params, try_params
#from defs.sgd import get_params, try_params
# from defs.keras_mlp import get_params, try_params
#from defs.polylearn_fm import get_params, try_params
#from defs.polylearn_pn import get_params, try_params
#from defs.xgb import get_params, try_params
#from defs.meta import get_params, try_params
from lda import get_params, try_params

'''
#--------------------------------------------------random search
try:
    output_file_RS = sys.argv[1]
    if not output_file_RS.endswith( '.pkl' ):
        output_file_RS += '.pkl'    
except IndexError:
    output_file_RS = 'results_RS.pkl'
    
print("RS Will save results to", output_file_RS)

hb2 = Hyperband( get_params, try_params )
results_RS = hb2.run2( skip_last = 0)

print("{} total, best:\n".format(len(results_RS)))

for r in sorted( results_RS, key = lambda x: x['loss'] )[:10]:
    print( "loss: {:.2%} |auc: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format( 
        r['loss'],r['auc'], r['seconds'], r['iterations'], r['counter'] ))
    pprint( r['params'] )
    print

print ("saving...")

with open( output_file_RS, 'wb' ) as f:
    pickle.dump( results_RS, f )
   
staRS_loss=[sub['best_loss'] for sub in results_RS]
staRS_sec=[sub['seconds'] for sub in results_RS]

'''

'''
#-------------------------------------------------- hyperband / iteration-based
try:
    output_file = sys.argv[1]
    if not output_file.endswith( '.pkl' ):
        output_file += '.pkl'   
except IndexError:
    output_file = 'results_HB.pkl'
    
print("HB Will save results to", output_file)

hb = Hyperband(get_params, try_params)
results = hb.run(skip_last = 0)

print("{} total, best:\n".format(len(results)))

for r in sorted( results, key = lambda x: x['loss'] )[:10]:
    print( "loss: {:.2%} |auc: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format( 
                 r['loss'], r['auc'], r['seconds'], r['iterations'], r['counter']))
    pprint(r['params'])
    print()

print ("saving results...")

with open(output_file, 'wb') as f:
    pickle.dump(results, f)

sta_loss=[sub['best_loss'] for sub in results]
sta_sec=[sub['seconds'] for sub in results]

score = sta_loss
runtime = sta_sec
with open('score_vs_time.csv','w') as f_score:
    f_score.write('Score,Time(s)\n')
    for i in range(len(score)):
        f_score.write('{},{}'.format(score[i], runtime[i]))
'''

#-------------------------------------------------- hyperband / Data-based for LDA

try:
    output_file = sys.argv[2]
    if not output_file.endswith( '.pkl' ):
        output_file += '.pkl'   
except IndexError:
    output_file = 'results_HB_LDA.pkl'
    
print("HB will save results to", output_file)

emb_model = sys.argv[1]

hb = Hyperband_LDA(get_params, try_params, 556, emb_model)
# results = hb.run(dry_run=True)
results = hb.run()

print("\n---------------- {} total, best 10 are:\n".format(len(results)))

for r in sorted(results, key = lambda x: x['lda_score'], reverse=True)[:10]:
    print( "lda_score: {:.4} | {} seconds | {:.1f} n_doc | run {} ".format( 
                 r['lda_score'], r['seconds'], r['n_doc'], r['counter']))
    pprint(r['params'])
    print()

print ("saving results ...")
with open(output_file, 'wb') as f:
    pickle.dump(results, f)

sta_loss=[sub['best_score'] for sub in results]
sta_sec=[sub['seconds'] for sub in results]

score = sta_loss
runtime = sta_sec
print('saving score_vs_time.csv ...')
with open('score_vs_time.csv','w') as f_score:
    f_score.write('Score,Time(s)\n')
    for i in range(len(score)):
        f_score.write('{},{}\n'.format(score[i], runtime[i]))

print("\nAuto-LDA finished.")

