import pandas as pd
from dataset import read_file, count_by_theme

df = pd.read_csv('data.tsv', sep='\t', index_col='text_id', quoting=3)[['text']]

raw_df = read_file('mydata/labelled/results_20210619121105.tsv')
raw_df['p_opinion'] = 0

cols = ['pos', 'neg', 'neut', 'posneg', 'impos', 'irrel']
for t in ['masks', 'vaccines', 'quarantine', 'government']:
    counts = count_by_theme(raw_df, t)
    counts.drop(columns=['p_opinion'], inplace=True)
    counts = counts[counts[['irrel', 'rel']].sum(axis=1) >= 2]

    relevant = counts[counts.rel >= 2]
    positive = relevant[(relevant.pos > relevant[filter(lambda x: x != 'pos', cols)].max(axis=1))]
    negative = relevant[(relevant.neg > relevant[filter(lambda x: x != 'neg', cols)].max(axis=1))]
    assert positive.index.intersection(negative.index).size == 0
    others = relevant.drop(positive.index.union(negative.index))

    df.loc[counts.index, t] = -1
    df.loc[negative.index, t] = 0
    df.loc[others.index, t] = 1
    df.loc[positive.index, t] = 2
    df.dropna(inplace=True)
    df[t] = df[t].astype(int)

df.to_csv('mydata/labelled/data.tsv', quoting=3, sep='\t')
pass