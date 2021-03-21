"""
Separates sentences to themes by finding keywords.
"""


import pandas as pd
from data_preparation import tokenize, lemmatize
from multiprocessing import Pool
import numpy as np


themes = {
        'vaccine': {'вакцина', 'вакцинный', 'вакцинация', 'иммунизация', 'вакцинировать', 'вакцинирование', 'прививка',
                    "прививать", "прививочный", "спутник", "спутник v", "moderna", "pfizer", "ковивак", "эпиваккорона", "astrazeneca"
        },
        'masks': {'маска', 'масочный'},
        'lockdown': {'карантин', 'карантинный', 'локдаун'},
        'covid': {"коронавирус", "ковид", "ковид19", "ковид-19", "covid", 'covid-19', 'covid19', "корона"}
        }


def process_line(line, keywords):
    words = tokenize(line)
    words_ = list(map(lambda x: lemmatize(x).lower(), words))
    for ngram in range(1, 2):
        pos = 0
        for i, w in enumerate(words_):
            if i + ngram > len(words_):
                break
            if words[i] in ["''", "``"]:
                words[i] = '"'
            pos = line.find(words[i], pos)
            end = line.find(words[i + ngram - 1], pos) + len(words[i + ngram - 1])
            candidate = ' '.join(words_[i: i + ngram])
            if candidate in keywords:
                is_absent = 1 if {"нет", "не", "ни", "без"}.intersection(words_[i - 2: i]) else 0
                if is_absent == 0 and i + ngram < len(words_):
                    is_absent = int(words_[i + ngram] == 'нет')
                # print(line)
                # print(line[pos: end])
                # print(words)
                # print()

                return True, is_absent, pos, end
            pos += len(words[i])
    return False, None, None, None


def check_row(row, kwords):
    return process_line(row[1], kwords)


if __name__ == '__main__':
    df = pd.read_csv('mydata/RIA/opinion classified/ria_scores_by_user.tsv', index_col=None, header=None, sep='\t', quoting=3)

    for theme, words in themes.items():
        p = Pool(10)
        m = p.starmap(check_row, zip(df.values, [words] * df.shape[0]), )
        idx, absence, *boundaries = zip(*m)
        p.close()
        # exit(0)
        boundaries = list(zip(*boundaries))

        # print(np.count_nonzero(m))
        df['absence'] = absence
        df['bounds'] = boundaries

        df.rename(columns={0: "usr_id", 1: "text", 2: "p_opinion", 3: "is_opinion"}, inplace=True)
        found = df[list(idx)]
        print(theme, found.shape)
        found.to_csv(f'mydata/ria_{theme}.tsv', sep='\t', header=False, index=False, quoting=3)
