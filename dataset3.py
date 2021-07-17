import numpy as np
import pandas as pd

train = pd.read_csv(f'mydata/labelled/train.tsv', index_col=['text_id'], quoting=3, sep='\t')
test = pd.read_csv(f'mydata/labelled/test.tsv', index_col=['text_id'], quoting=3, sep='\t')

for t in ['masks', 'vaccines', 'quarantine', 'government']:
    train['relevant'] = (train[t] >= 0).astype(int)
    test['relevant'] = (test[t] >= 0).astype(int)

    train[['text', 'relevant']].to_csv(f'mydata/labelled/{t}/{t}_rel_train.tsv', sep='\t', quoting=3)
    test[['text', 'relevant']].to_csv(f'mydata/labelled/{t}/{t}_rel_test.tsv', sep='\t', quoting=3)

    tr = train[train[t] >= 0]
    te = test[test[t] >= 0]

    tr[['text', t]].rename(columns={t: 'sen'}).to_csv(f'mydata/labelled/{t}/{t}_sen_train.tsv', sep='\t', quoting=3)
    te[['text', t]].rename(columns={t: 'sen'}).to_csv(f'mydata/labelled/{t}/{t}_sen_test.tsv', sep='\t', quoting=3)

