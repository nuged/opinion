"""
Calculates contrast between comments from different weeks
"""

from contrast import weirdness, write_results
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.corpus import stopwords
from string import punctuation
from multiprocessing import Pool
import pymorphy2

stop_words = stopwords.words('russian')
stop_words += ['``', "''", '—', '«', '»', '...', '–']
morph = pymorphy2.MorphAnalyzer()


def read(file):
    with open(file) as f:
        texts = f.read().split('\n')[:-1]
    return texts


def tokenize(text):
    if '[USER]' in text:
        words = ['[USER]', ','] + tokenize(text[7:])
    else:
        words = word_tokenize(text, language='russian')
    return words


def lemmatize(word):
    return morph.parse(word)[0].normal_form


def get_words(text):
    words = tokenize(text.lower())
    result = []
    for w in words:
        w = lemmatize(w)
        if w not in stop_words and w not in punctuation and not w.isdigit():
            result.append(w)
    return result


def count_words(texts):
    wc = defaultdict(int)
    with Pool(10) as p:
        words = p.map(get_words, texts)
    for doc in words:
        for w in doc:
            wc[w] += 1
    result = {}
    for word, count in wc.items():
        if count >= 5:
            result[word] = count
    return result


data = [read(f'mydata/ria_comments_{week}.txt') for week in [1, 2, 3]]
counts = [count_words(data[i]) for i in [0, 1, 2]]

for w1 in [1, 2, 3]:
    for w2 in [1, 2, 3]:
        if w2 > w1:
            scores = weirdness(counts[w1 - 1], counts[w2 - 1], min_freq=5)
            write_results(f'mydata/ria_weirdness_{w2}_{w1}.txt', scores)
