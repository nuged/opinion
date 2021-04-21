"""
Separates sentences to themes by finding keywords.
"""


import pandas as pd
from data_preparation import tokenize, lemmatize
from multiprocessing import Pool
import numpy as np
from  itertools import product
import pymorphy2
import time

themes = {
        'vaccine': {'вакцина', 'вакцинный', 'вакцинация', 'иммунизация', 'вакцинировать', 'вакцинирование', 'прививка',
                    "прививать", "прививочный", "спутник", "спутник v", "moderna", "pfizer", "ковивак", "эпиваккорона", "astrazeneca"
        },
        'masks': {'маска', 'масочный'},
        'lockdown': {'карантин', 'карантинный', 'локдаун'},
        'covid': {"коронавирус", "ковид", "ковид19", "ковид-19", "covid", 'covid-19', 'covid19', "корона", "sars-cov-2"}
        }

morph = pymorphy2.MorphAnalyzer()

def process_line(line, keywords):
    words = tokenize(line)
    line = line.lower()
    words = list(map(str.lower, words))
    # words_ = list(map(lambda x: lemmatize(x).lower(), words))
    for i, w in enumerate(words):
        if words[i] in ["''", "``"]:
            words[i] = '"'
    result = []
    for ngram in range(1, 2):
        pos = 0
        for i, w in enumerate(words):
            if i + ngram > len(words):
                break

            pos = line.find(words[i], pos)
            assert pos != -1

            end = line.find(words[i + ngram - 1], pos)
            assert end != -1

            end += len(words[i + ngram - 1])

            assert "".join(words[i: i + ngram]) == line[pos: end]

            normals = []
            for word in words[i: i + ngram]:
                normals.append([x.normal_form for x in morph.parse(word) if x.score > 0.01])
            normals = product(*normals)

            for c in normals:
                candidate = ' '.join(c)
                if candidate in keywords:
                    prefix = [morph.parse(word)[0].normal_form for word in words[i - 2: i]]
                    is_absent = 1 if {"нет", "ни", "без", "нехватка", "нету",
                                      "дефицит", "отсутствие"}.intersection(prefix) else 0
                    is_absent = 0 if prefix and prefix[-1] in {',', ';', '...'} else is_absent
                    if is_absent == 0 and i + ngram < len(words):
                        is_absent = int(words[i + ngram] in ['нет', 'нету'])
                    result.append((is_absent, pos, end))
                    break

            pos += len(words[i])
    return result


if __name__ == '__main__':
    from data_preparation import fix_quotes
    df = pd.read_csv('mydata/citations/opinion_scores.tsv', index_col=None, header=None, sep='\t', quoting=3)
    df[0] = df[0].apply(fix_quotes)
    print(df.shape)
    for theme, words in themes.items():
        p = Pool(10)
        m = p.starmap(process_line, zip(df[0].values, [words] * df.shape[0]), chunksize=10000)
        p.close()
        idx = [bool(item) for item in m]
        found = df[idx]
        absence = [[elem[0] for elem in item] for item in m if item]
        bounds = [[elem[1:] for elem in item] for item in m if item]

        # print(np.count_nonzero(m))
        found['absence'] = absence
        found['bounds'] = bounds

        found.rename(columns={0: "text", 1: "p_opinion"}, inplace=True)
        print(theme, found.shape)
        found.to_csv(f'mydata/citations/{theme}.tsv', sep='\t', index=False, quoting=3)
