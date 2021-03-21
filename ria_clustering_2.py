"""
Clustering with lexical and syntactical features
"""

import pandas as pd
from ria_themes import process_line, themes
import numpy as np
from data_preparation import lemmatize, tokenize, stopwords, punctuation
from sklearn.metrics import f1_score, precision_score, recall_score

df = pd.read_csv('mydata/RIA/topics/scored data/ria_vaccine_labelled.tsv', index_col=False, quoting=3, sep='\t')
pd.set_option("display.max_rows", None, "display.max_columns", None)
labels = {'-': 0, '=': 1, '+': 2}
labelled = list(map(lambda x: labels[x[0]], df['usr_id'].values))
df['Topic'] = labelled


tmp = np.array(df['text'].map(lambda x: process_line(x, themes['vaccine'])[2:]).tolist())
df['start'] = tmp[:, 0]
df['end'] = tmp[:, 1]

absence = {"нет", "не", "ни", "без"}
neg_ = set()
pos_ = set()
with open('opinion/rusentilex_2017.txt') as f:
    for line in f:
        if line.startswith('!') or ', ' not in line:
            continue
        _, pos, lemma, sent, *_ = line.split(', ')
        if sent == 'positive':
            pos_.add(lemma)
        elif sent == 'negative':
            neg_.add(lemma)


def is_close(row):
    text = row.text[: row.start]
    words = tokenize(text)[-10:]
    words = list(map(lemmatize, words))
    if not words:
        return 0
    if words[-1] in absence:
        words = words[: -1]
    # words = list(filter(lambda x: x not in punctuation and x not in stopwords, words))
    if not words:
        return 0
    previous = words[-1]
    negated = not len(words) < 2 and words[-2] in ['не', "нет", "ни"]
    if previous in pos_:
        score = 1
    elif previous in neg_:
        score = -1
    else:
        score = 0
    score = -score if negated else score
    score = -score if row.absence else score
    return score


for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    neutral = df[(df.p_opinion < t) & (df.sent_words == '[]')]
    y_pred = np.zeros(df.shape[0])
    y_pred[neutral.index] = 1
    y_true = (df.Topic == 1)
    f1 = f1_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f'{t}:\tf1={f1:4.3f}, pr={pr:4.3f}, rec={rec:4.3f}, size={neutral.shape[0]}')


negative = df[df.apply(is_close, axis=1) == -1]
positive = df[df.apply(is_close, axis=1) == 1]
threshold = 0.2
neutral = df[(df.p_opinion < threshold) & (df.sent_words == '[]')]

neutral.drop(columns=['start', 'end']).to_csv(f'mydata/neutral.tsv', sep='\t', header=True, index=False, quoting=3)
positive.drop(columns=['start', 'end']).to_csv(f'mydata/positive.tsv', sep='\t', header=True, index=False, quoting=3)
negative.drop(columns=['start', 'end']).to_csv(f'mydata/negative.tsv', sep='\t', header=True, index=False, quoting=3)

df['test'] = -1
df['test'][negative.index] = 0
df['test'][neutral.index] = 1
df['test'][positive.index] = 2

df.drop(columns=['start', 'end']).to_csv('mydata/clustering_test.tsv', sep='\t', header=True, index=False, quoting=3)
