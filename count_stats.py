import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split


def get_theme(row, col_name):
    if 'вакцин' in row[col_name].lower():
        return "Вакцинирование"
    if 'карант' in row[col_name].lower():
        return "Карантин"
    if 'маск' in row[col_name].lower():
        return 'Маски'
    if 'правит' in row[col_name].lower():
        return 'Правительство'


def get_stats(filename, answers):
    df = pd.read_csv(filename, quoting=3, sep='\t')
    col_name = 'hypothesis' if 'nli' in filename else 'question'
    average = None # 'binary' if 'nli' in filename else 'macro'
    data = df[['text', col_name]]
    data['theme'] = data.apply(lambda row: get_theme(row, col_name), axis=1)

    data = data[['text', 'theme']]
    labels = df.label

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, random_state=35,
                                            stratify=labels)

    with open(answers, 'rb') as f:
        y_true, y_pred = pickle.load(f)
        assert (y_true == y_test).all()

    counts = pd.DataFrame()
    counts['train_pct'] = X_train.groupby('theme').size()
    counts['test_pct'] = X_test.groupby('theme').size()
    counts['n_total'] = data.groupby('theme').size()
    counts.train_pct /= counts.n_total / 100
    counts.test_pct /= counts.n_total / 100
    counts = counts.round(2)

    X_test['y_true'] = y_test
    X_test['y_pred'] = y_pred

    scores = pd.DataFrame(columns=['acc', 'pr', 're', 'f1'])
    for name, g in X_test.groupby('theme'):
        ac = accuracy_score(g.y_true, g.y_pred) * 100
        pr = precision_score(g.y_true, g.y_pred, average=average) * 100
        re = recall_score(g.y_true, g.y_pred, average=average) * 100
        f1 = f1_score(g.y_true, g.y_pred, average=average) * 100
        scores = scores.append(pd.Series([ac, pr, re, f1], name=name, index=['acc', 'pr', 're', 'f1']))

    ac = accuracy_score(X_test.y_true, X_test.y_pred) * 100
    pr = precision_score(X_test.y_true, X_test.y_pred, average=average) * 100
    re = recall_score(X_test.y_true, X_test.y_pred, average=average) * 100
    f1 = f1_score(X_test.y_true, X_test.y_pred, average=average) * 100
    scores = scores.append(pd.Series([ac, pr, re, f1], name='Total', index=['acc', 'pr', 're', 'f1']))

    scores = scores.round(2)

    return counts, scores


if __name__ == '__main__':
    for task in ['nli', 'qa']:
        for set in ['relevance', 'sentiment']:
            if task == 'qa' and set == 'relevance':
                continue
            counts, scores = get_stats(filename=f'mydata/labelled/{set}_{task}.tsv',
                                        answers=f'mydata/labelled/res/{set}_{task.upper()}_RuBERT_conv.pk')
            counts.to_csv(f'mydata/labelled/res/{set}_{task}.tsv', sep='\t', quoting=3)
            scores.to_csv(f'mydata/labelled/res/{set}_{task}.tsv', sep='\t', quoting=3, mode='a')

    for t in ['masks', 'quarantine', 'government', 'vaccines']:
        for task in ['relevance', 'sentiment']:
            path = f'mydata/labelled/singles/{task}_{t}_single_RuBERT_conv.pk'
            with open(path, 'rb') as f:
                y_true, y_pred = pickle.load(f)
            g = open(f'mydata/labelled/{task}/{t}_comparison.tsv', 'a')
            print(f"{task}\t{t}\t{accuracy_score(y_true, y_pred) * 100}\t"
                  f"{precision_score(y_true, y_pred, average=None) * 100}\t"
                  f"{recall_score(y_true, y_pred, average=None) * 100}\t"
                  f"{f1_score(y_true, y_pred, average=None) * 100}\t")
            g.close()
