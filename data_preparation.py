from nltk.corpus import stopwords
from multiprocessing import Pool
from nltk import word_tokenize
from unicodedata import normalize
from string import punctuation
import numpy as np
import pymorphy2
import re


stopwords = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()
punctuation += "«»—–"
punctuation = list(punctuation) + ["``", "''"]


def replace_numbers(text):
    pattern = re.compile(r"(?<![\d\w-])-?\+?[\d,\s:.]*\d+[\-\d,\s:.]*(?![\d\w-])")
    return pattern.sub(" -num- ", text)


def remove_links(text):
    pattern = re.compile(r"https?:.+\b")
    text = pattern.sub(r'', text)
    return re.sub(r'\w+\.\w\w\w?', '', text)


def remove_emoji(text):
    pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\u200d\u200b"
        u"\U0001F900-\U0001F9FF"
                           "]+", flags = re.UNICODE)
    return pattern.sub(r'',text)


def tokenize(x):
    return word_tokenize(x, language="russian")


def lemmatize(w):
    return morph.parse(w)[0].normal_form


def read_data(filename):
    with open(filename) as f:
        data = f.read().split("\n")
    data = map(lambda x: normalize("NFKC", x), data)
    data = list(map(lambda x: x.strip(), data))
    # data = list(filter(lambda x: x, data))
    return data


def remove_duplicates(data):
    """
    WORKS WITH ONLY ENUMERATED DATA
    :param data:
    :return:
    """
    unique = set()
    result = []
    for d in data:
        if isinstance(d[1], list):
            sd = ' '.join(d[1])
        else:
            sd = d[1]
        if sd not in unique:
            unique.add(sd)
            result.append(d)
    return result


def process_chunk(text, replace_nums=True):
    text = remove_links(text)
    text = remove_emoji(text)
    if replace_nums:
        text = replace_numbers(text)
    text = tokenize(text)
    text = map(lemmatize, text)
    text = filter(lambda x: x not in stopwords and x not in punctuation, text)
    return list(text)


def process_data(data, num_processes=12, chunk_size=64):
    p = Pool(num_processes)
    res = p.map(process_chunk, data, chunksize=chunk_size)
    p.close()
    return res


def remove_short(data):
    return list(filter(lambda x: len(x[1]) > 4, data))


def load(filename):
    data = read_data(filename)
    data = process_data(data)
    data = enumerate(data)
    data = remove_duplicates(data)
    data = remove_short(data)
    print(f"{len(data)} sentences were loaded from {filename}")
    ids, data = list(zip(*data))
    return ids, data


def make_dataset(posfile, negfile):
    pos = load(posfile)[1]
    neg = load(negfile)[1]
    pos = list(pos)
    labels = [+1] * len(pos) + [-1] * len(neg)
    pos.extend(neg)
    return pos, np.array(labels)


def choose(infile, outfile, idx):
    with open(infile) as f, open(outfile, "w") as g:
        for i, line in enumerate(f):
            if i in idx:
                g.write(line)


if __name__ == "__main__":
    for file in ['data/pos.txt', 'data/neg.txt']:
        ids, _ = load(file)
        print(len(ids))
        print(_[:5])
        print(_[-5:])
        choose(file, file[5:-4] + '_c.txt', ids)

# Единичные ФАКТЫ в Чамзинском и Лямбирском районах.