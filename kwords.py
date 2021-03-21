import re
from data_preparation import tokenize, lemmatize
from multiprocessing import Pool
from itertools import repeat
import pandas as pd
from unicodedata import normalize
# from nltk.stem.snowball import SnowballStemmer
#
# stemmer = SnowballStemmer("russian")
# lemmatize = lambda word: stemmer.stem(word)

# lem: 578, stem: 564

def process_line(line, keywords):
    words = tokenize(line)
    found = []
    words_ = list(map(lambda x: lemmatize(x).lower(), words))
    for ngram in range(1, 11):
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
                found.append((line[pos: end], pos, end))
            pos += len(words[i])
    return line.strip(), found


if __name__ == '__main__':
    keywords = set()
    with open("data/rubrs.json") as f:
        for line in f:
            line = re.sub(r'\\', '', line)
            entries = re.findall(r'"textentrystr":\s"(.+?)",\s"confirmid"', line)
            entries = list(map(tokenize, entries))
            entries = [[lemmatize(t) for t in tokens] for tokens in entries]
            entries = map(' '.join, entries)
            entries = map(str.lower, entries)
            entries = list(entries)
            if entries:
                keywords.update(entries)

    data = pd.read_excel("Coronavirus.xlsx", engine='openpyxl')[['Ссылка', 'Высказывание', 'Ключевые слова', 'Счёт']].dropna()
    data = data[['Высказывание', "Ключевые слова"]].values
    data = list(map(lambda x: [normalize("NFKC", x[0]), x[1].lower().split(', ')], data))

    for i, d in enumerate(data):
        words = []
        for w in d[1]:
            cands = re.findall(r'(.*)\((.+)\)', w)
            if cands:
                cands = list(map(str.strip, cands[0]))
                cands = list(map(lemmatize, cands))
                words.extend(cands)
            else:
                w = tokenize(w)
                w = map(lemmatize, w)
                w = ' '.join(w)
                words.append(w)
        data[i][1] = words

    p = Pool(10)
    result = p.starmap(process_line, data)
    p.close()
    f = open('mydata/kwords/kwords_manual.tsv', 'w')
    g = open('mydata/kwords/kwords_auto.tsv', 'w')
    failed = 0
    for i, r in enumerate(result):
        if r[1]:
            print(f"{r[0]}\t{r[1]}", file=f)
        else:
            r_ = process_line(r[0], keywords)
            if r_[1]:
                print(f"{r_[0]}\t{r_[1]}", file=g)
            else:
                failed += 1
                print(data[i])
    f.close()
    g.close()

    print(failed)

