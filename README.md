# AutoLDA

Load pretrained embedding models:

1. For GLOVE, download the pretrained model to folder ./Embeddings/GLOVE_pretrained/, then run GLOVE.py to save the loaded model to pkl file

```
$ wget http://nlp.stanford.edu/data/glove.840B.300d.zip
$ unzip glove.840B.300d.zip 
$ python GLOVE.py 
```

2. To run hyperband with LDA:
```console
$ python main.py results_hb_glove.pkl GLOVE
$ python main.py results_hb_bert.pkl BERT
```

3. To show the best/worst 10 configurations:
```console
$ python python show_results.py results_hb_glove.pkl 10
$ python python show_results.py results_hb_glove.pkl -10
```

4. To run the best and worst 10 configs on full data and print out topic_keywords:
```console
$ python run_configs.py 10
```