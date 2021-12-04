#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 00:35:52 2021

@author: tian
"""

import matplotlib.pyplot as plt
import pandas as pd


# W2V

# data = pd.read_csv('score_vs_time_W2V.csv')
# data = pd.read_csv('score_vs_time_GLOVE.csv')
data = pd.read_csv('score_vs_time_BERT.csv')

x=data['Time(s)']
y=data['score']

fig, ax = plt.subplots(figsize=(6,6))

# ax.plot(x, y, label='HB-W2V', marker='o', color='red')
# ax.plot(x, y, label='HB-GLOVE', marker='o', color='red')
ax.plot(x, y, label='HB-BERT', marker='o', color='red')

plt.legend()
plt.xlim([0,1400])
plt.ylim([0.2,0.6])

ax.set(xlabel='Time (s)', ylabel='Embedding_Score', title='')
# plt.xticks(np.arange(0, 50, step=5))
# ax.grid()

# fig.savefig("test.png")
plt.show()