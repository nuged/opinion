"""
Evaluates sentiments with lexical features for each theme.
"""


import pandas as pd
from data_preparation import tokenize, stopwords, punctuation, lemmatize
from multiprocessing import Pool
from ria_themes import themes
import pymorphy2
from itertools import product
from math import *

sentiments = {}
morph = pymorphy2.MorphAnalyzer()

with open('opinion/rusentilex_2017.txt') as f:
    for line in f:
        if line.startswith('!') or ', ' not in line:
            continue
        _, pos, lemma, sent, *_ = line.split(', ')
        if sent == 'positive':
            sent = 1
        elif sent == 'negative':
            sent = -1
        else:
            continue
        sentiments[lemma] = sent


def calc_distance(start, end, pos, words):
    assert(start <= end)

    if start > pos:
        left = pos + 1
        right = start
    if end < pos:
        left = end + 1
        right = pos
    subwords = words[left: right]
    subwords = list(map(lemmatize, subwords))
    # print(subwords)
    subwords = list(filter(lambda x: x not in stopwords and x not in punctuation, subwords))
    # print(subwords)
    # print(words)
    # print(words[start: end + 1])
    # print(words[pos], len(subwords) + 1)
    # print()
    return len(subwords) + 1

def process_line(line, bounds, keywords):
    words = tokenize(line)
    theme_word = line[bounds[0]: bounds[1]]
    theme_word_pos = words.index(theme_word)
    found = []
    total_score = 0
    # words_ = list(map(lambda x: lemmatize(x).lower(), words))
    for ngram in range(1, 4):
        pos = 0
        for i, w in enumerate(words):
            if i + ngram > len(words):
                break
            if words[i] in ["''", "``"]:
                words[i] = '"'
            pos = line.find(words[i], pos)
            end = line.find(words[i + ngram - 1], pos) + len(words[i + ngram - 1])
            normals = []
            for word in words[i: i + ngram]:
                normals.append([x.normal_form for x in morph.parse(word) if x.score > 0.01
                                and word.lower() not in ['уже', "какая", "толком", "просто"]])
            normals = product(*normals)
            prev_word = morph.parse(words[i - 1])[0].normal_form
            is_negated = prev_word in ['не', "нет", "ни"]
            for c in normals:
                candidate = ' '.join(c)
                if candidate in keywords:
                    start_w = i - 1 if is_negated else i
                    end_w = i + ngram - 1
                    distance = calc_distance(start_w, end_w, theme_word_pos, words)
                    if distance > 10:
                        continue
                    score = sentiments[candidate] * (1 - log10(distance))
                    score = -score if is_negated else score
                    candidate = prev_word + ' ' + candidate if is_negated else candidate
                    p = pos - 3 if is_negated else pos
                    total_score += score
                    found.append((candidate, p, end))
                    break
            pos += len(words[i])
    return line.strip(), found, total_score


def score_line(line):
    line, bounds = line
    bounds = eval(bounds)
    _, found, score = process_line(line, bounds, sentiments)
    return score, found


if __name__ == '__main__':
    for theme in themes:
        df = pd.read_csv(f'mydata/ria_{theme}.tsv', index_col=None, header=None, sep='\t', quoting=3)

        g = df.groupby(0)
        # df = g.filter(lambda x: x[1].count() > 1)
        print(theme, df.shape[0])
        with Pool(10) as p:
            scores = p.map(score_line, df.values[:, [1, -1]])
            scores = list(zip(*scores))
            # print(scores)
            ncols = df.shape[1]
            df[ncols], df[ncols + 1] = scores

        df.drop(columns=[5], inplace=True)

        df.rename(columns={0: 'usr_id', 1: 'text', 2: 'p_opinion', 3: 'is_opinion', 4: 'absence', 6: 'sentiment',
                           7: 'sent_words'}, inplace=True)

        df.to_csv(f'mydata/ria_{theme}_scores.tsv', sep='\t', header=True, index=False, quoting=3)

# g = df.groupby(0)
# pos = g.filter(lambda x: (x[4] > 0).all())
# neg = g.filter(lambda x: (x[4] < 0).all())
#
# pos.to_csv('mydata/ria_usr_vac_scores_pos.tsv', sep='\t', header=False, index=False, )
# neg.to_csv('mydata/ria_usr_vac_scores_neg.tsv', sep='\t', header=False, index=False, )
