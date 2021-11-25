# AutoLDA

Load pretrained embedding models:

1. For GLOVE, download the pretrained model to folder ./Embeddings/GLOVE_pretrained/, then run GLOVE.py to save the loaded model to pkl file

$ wget http://nlp.stanford.edu/data/glove.840B.300d.zip
$ unzip glove.840B.300d.zip 
$ python GLOVE.py 

2. To run hyperband with LDA:
$ python main.py GLOVE results_HB_glove.pkl
