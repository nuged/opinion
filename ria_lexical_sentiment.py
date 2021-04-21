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
        line = line.strip()
        _, pos, lemma, sent, source, *_ = line.split(', ')
        if source not in ['fact', 'feeling', 'opinion', 'operator']:
            continue
        if sent == 'positive':
            sent = 1
        elif sent == 'negative' and source != 'fact':
            sent = -1
        else:
            continue
        sentiments[lemma] = sent


def calc_distance(start_idx, end_idx, kw_ids, words):
    # print(words)
    # print(start_idx, end_idx, kw_ids)
    assert(start_idx <= end_idx)
    min_dist = len(words) + 1
    nearest_kw = None
    for i, idx in enumerate(kw_ids):
        if start_idx > idx:
            left = idx + 1
            right = start_idx
        if end_idx < idx:
            left = end_idx + 1
            right = idx
        subwords = words[left: right]
        subwords = list(map(lemmatize, subwords))
        subwords = list(filter(lambda x: x not in stopwords and x not in punctuation, subwords))
        if len(subwords) + 1 < min_dist:
            min_dist = len(subwords) + 1
            nearest_kw = i
    # print(nearest_kw, min_dist)
    # print()
    return nearest_kw, min_dist


def process_line(line, bounds, absences, keywords):
    words = tokenize(line)
    # line = line.lower()
    # words = list(map(str.lower, words))

    theme_words = [line[b[0]: b[1]] for b in bounds]
    assert all(theme_words)

    theme_word_ids = [words.index(word) for word in theme_words]

    found = []
    total_score = 0
    # words_ = list(map(lambda x: lemmatize(x).lower(), words))
    for i, w in enumerate(words):
        if words[i] in ["''", "``"]:
            words[i] = '"'

    for ngram in range(1, 4):
        pos = 0
        for i, w in enumerate(words):
            if i + ngram > len(words):
                break

            pos = line.find(words[i], pos)
            assert pos != -1

            p = pos
            for n in range(1, ngram):
                p = line.find(words[i + n - 1], p) + len(words[i + n - 1])
            # p = pos + sum(len(words[i + n - 1]) for n in range(1, ngram))
            end = line.find(words[i + ngram - 1], p)
            assert end != -1
            end += len(words[i + ngram - 1])
            tmp = line[pos: end].replace(' ', '')
            pp = "".join(words[i: i + ngram])
            assert pp == tmp

            normals = []
            for word in words[i: i + ngram]:
                normals.append([x.normal_form for x in morph.parse(word) if x.score > 0.01
                                and word.lower() not in ['уже', "какая", "толком", "просто"]])
            normals = product(*normals)

            prev_word = morph.parse(words[i - 1])[0].normal_form
            is_negated = prev_word in ['не', "нет", "ни", 'нету']

            for c in normals:
                candidate = ' '.join(c)
                if candidate in keywords:
                    start_idx = i - 1 if is_negated else i
                    end_idx = i + ngram - 1
                    nearest_kw, distance = calc_distance(start_idx, end_idx, theme_word_ids, words)
                    if distance > 10:
                        continue
                    score = sentiments[candidate] * (1 - log10(distance))
                    score = -score if is_negated else score
                    score = -score if absences[nearest_kw] == 1 else score
                    candidate = prev_word + ' ' + candidate if is_negated else candidate
                    p = pos - len(prev_word) - 1 if is_negated else pos
                    total_score += score
                    found.append((candidate, p, end))
                    break

            pos += len(words[i])
    return line.strip(), found, total_score


def score_line(line):
    line, bounds, absences = line
    bounds = eval(bounds)
    absences = eval(absences)
    _, found, score = process_line(line, bounds, absences, sentiments)
    return score, found


if __name__ == '__main__':
    for theme in themes:
        df = pd.read_csv(f'mydata/citations/{theme}.tsv', index_col=None, sep='\t', quoting=3)

        # g = df.groupby('usr_id')

        print(theme, df.shape[0])
        with Pool(10) as p:
            scores = p.map(score_line, df[['text', 'bounds', 'absence']].values, chunksize=1000)
        scores = list(zip(*scores))
        # print(scores)
        ncols = df.shape[1]
        df['sentiment'], df['sent_words'] = scores

        # df.drop(columns=['bounds'], inplace=True)

        df.to_csv(f'mydata/citations/{theme}_sentiment.tsv', sep='\t', header=True, index=False, quoting=3)
