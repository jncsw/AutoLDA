# AutoLDA

This repo implements the Hyperband with LDA, aiming to build a AutoLDA tool to find the best hyperparameters for LDA.

Different implementation schemes of Hyperband with LDA was implemented in different branches.

1. main: Hyperband with data (# of documents) as resources. This scheme allocates more training data to most promising configs. Tests shows this does not work well.
2. hyperband_iter: Hyperband with # of iterations as resouces, perplexity as metric to evaluate the goodness of LDA. This schemes splits data into training data and test data. The perplexity of test data is used as evaluation metric to filter the good configs.
3. iter_w2v: Hyperband with # of iterations as resources. This scheme uses full data to train the LDA with given iterations. Different embedding methods were used to calculate the embedding score.

Switch to different branches to try different implementation schemes.

## Hyperband + LDA
To run Hyperband with LDA:
```console
python main.py results_hb_W2V.pkl W2V
```

To show the best 10 configurations with its topic_words with given iterations:
```console
python show_results.py results_hb_W2V.pkl 10 W2V
```

To run the selected top1 config with full 81 resources to get the final LDA results:
```console
python run_configs.py 1 W2V
```

To plot the score vs. time of each embedding schemes:
```console
python plot.py
```


## Environment setup
Load pretrained embedding models:

1. For GLOVE, download the pretrained model to folder `./Embeddings/GLOVE_pretrained/`, then run `GLOVE.py` to save the loaded model to pkl file

```console
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
```
```console
unzip glove.840B.300d.zip 
```
```console
python GLOVE.py 
```
