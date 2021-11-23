

import os


import nltk
# import tensorflow as tf
import tensorflow_hub as hub
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

# pip uninstall tensorflow tensorboard tensorflow-estimator
# ...
# pip install tensorflow==1.14.0


# embeddings = elmo(
#     [
#         "I love to watch TV",
#         "I am wearing a wrist watch"
#     ],
#     signature="default",
#     as_dict=True)["elmo"]

init = tf.initialize_all_variables()
sess = tf.Session()


# # Print word embeddings for word WATCH in given two sentences
# print('Word embeddings for word WATCH in first sentence')
# print(sess.run(embeddings[0][3]))
# print('Word embeddings for word WATCH in second sentence')
# print(sess.run(embeddings[1][5]))


def genEmbeddings_ELMo(text):
    sess.run(init)
    embeddings = elmo(
    [
        text,
    ],
    signature="default",
    as_dict=True)["elmo"]




    return sess.run(embeddings[0][0])

if __name__ == "__main__":
    print("Embedding=",genEmbeddings_ELMo("watch"))
