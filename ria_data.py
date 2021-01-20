import pandas as pd
from data_preparation import *
from nltk.tokenize import sent_tokenize, TweetTokenizer
from multiprocessing import Pool
from random import sample


def norm(text):
    return normalize("NFKC", text)


def sent_tok(text):
    return sent_tokenize(text, language='russian')


def preprocess_data(path):
    comments = pd.read_csv(path, index_col=None)
    comments = comments.text.dropna().tolist()
    data = comments

    p = Pool(10)
    data = p.map(norm, data)
    data = p.map(fix_newlines, data)
    data = p.map(remove_links, data)
    data = p.map(remove_emoji, data)
    data = p.map(remove_vkid, data)
    data = p.map(fix_sentences, data)
    data = p.map(str.strip, data)
    data = p.map(sent_tok, data)
    data = [s for d in data for s in d if s and s != 'Deleted comment']
    tokenized_data = p.map(tokenize, data)
    tokenized_data = p.map(lemm_sent, tokenized_data)
    tokenized_data = [list(filter(lambda x: x not in punctuation and x not in stopwords, tokens))
                      for tokens in tokenized_data]
    p.close()

    data = [s for i, s in enumerate(data) if len(tokenized_data[i]) >= 5]
    print(len(data))
    data = set(data)
    print(len(data))
    return data


if __name__ == '__main__':
    for week in ['1', '2', '3']:
        data = preprocess_data(f'data/ria/ria_comments_{week}.csv')
        f = open(f'mydata/ria_comments_{week}.txt', 'w')
        for d in data:
            print(d, file=f)
        f.close()
