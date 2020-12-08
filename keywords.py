import pandas as pd
import pymorphy2
import random
import re
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from nltk.stem.snowball import SnowballStemmer


def spit_words(line, sep=','):
    pattern = re.compile(r"\((\w+)\)")
    line = pattern.sub(r", \g<1>", line)
    words = line.split(sep)
    words = map(str.strip, words)
    words = list(filter(lambda x: x, words))
    return words


def collect_keywords(df):
    result = set()
    for i, row in df.iterrows():
        kw = row['Ключевые слова']
        kw = spit_words(kw)
        for word in kw:
            result.add(word)
    return result

stemmer = SnowballStemmer("russian")
stem = lambda word: stemmer.stem(word.lower())
morph = pymorphy2.MorphAnalyzer()
# norm = lambda word: morph.parse(word.lower())[0].normal_form

def label(tokens, keywords):
    tokens.append("[PAD]")
    for i, kw in enumerate(keywords):
        kws = kw.split()
        kws = map(stem, kws)
        kws = " ".join(kws)
        keywords[i] = kws.lower()

    word = None
    start = None
    bigram = []
    result = []
    for i, t in enumerate(tokens):
        if not t.startswith("##"):
            if word is not None:
                word = "".join(word)
                if len(bigram) == 0:
                    bistart = start
                    bigram = [stem(word)]
                elif len(bigram) == 1:
                    bigram.append(stem(word))
                    bigram = " ".join(bigram)
                    if bigram in keywords:
                        result.append((bistart, i - 1))
                    bigram = [stem(word)]
                    bistart = start
                if stem(word) in keywords:
                    result.append((start, i - 1))
            word = [t]
            start = i
        else:
            word.append(t[2:])
    tokens.pop()
    return result


class MyDataset(Dataset):
    def __init__(self, df, tok):
        super(MyDataset, self).__init__()
        self.data = df
        self.keywords = collect_keywords(self.data)
        self.tokenizer = tok

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        sampled_kw = random.sample(self.keywords, k=3)
        query = "Найди такие слова, как {}, или {}, или {}.".format(*sampled_kw)
        query = self.tokenizer.tokenize(query)
        context = self.data.iloc[item]['Высказывание']
        # context = self.tokenizer.tokenize(context)
        keywords = self.data.iloc[item]['Ключевые слова']
        keywords = spit_words(keywords)
        labels = label(self.tokenizer.tokenize(context), keywords.copy())
        return context, query, labels, keywords


if __name__ == "__main__":
    data = pd.read_excel("data/Coronavirus.xlsx", )[['Ссылка', 'Высказывание', 'Ключевые слова', 'Счёт']].dropna()
    random.seed(5)
    ds = MyDataset(data, AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", do_lower_case=False))
    counter = 0
    for kw in ds:
        c, q, l, k = kw
        if len(l) >= 1:
            counter += 1
            print(c)
            print(k)
            print()

print(counter)

#539
#552