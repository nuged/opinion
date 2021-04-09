"""
Preprocesses comments and writes sentences to a file.
"""

import pandas as pd
from data_preparation import *
from nltk.tokenize import sent_tokenize, TweetTokenizer
from multiprocessing import Pool
from random import sample
from math import isnan

def norm(text):
    return normalize("NFKC", text)


def sent_tok(text):
    return sent_tokenize(text, language='russian')


def preprocess_data(path):
    comments = pd.read_csv(path, index_col=None, sep='\t', quoting=3, header=None)
    # users_na = comments.commenter_id.tolist()
    data = comments[0].tolist()
    # data = []
    # users = []
    # for i, comm in enumerate(comments):
    #     if isinstance(comm, str) and not isnan(users_na[i]):
    #         data.append(comm)
    #         users.append(users_na[i])

    p = Pool(10)
    data = p.map(norm, data)  # unicode normalization
    data = p.map(fix_newlines, data)  # replace newlines inside comments with a space
    data = p.map(remove_links, data)  # remove urls, emails and mentions
    data = p.map(remove_emoji, data)
    data = p.map(remove_vkid, data)  # replace user info with a special token in replies
    data = p.map(fix_sentences, data)  # fix spacing between sentences and near commas
    data = p.map(str.strip, data)  # remove extra spaces
    data = p.map(sent_tok, data)
    data = [sent for item in data for sent in item]
    # sents2users = {s: users[i] for i, d in enumerate(data) for s in d if s and s != 'Deleted comment'}
    # data = sents2users.keys()
    tokenized_data = p.map(tokenize, data)
    tokenized_data = p.map(lemm_sent, tokenized_data, chunksize=1000)
    tokenized_data = [list(filter(lambda x: x not in punctuation and x not in stopwords, tokens))
                      for tokens in tokenized_data]
    p.close()

    data = [s for i, s in enumerate(data) if len(tokenized_data[i]) >= 5]
    print(len(data))
    data = set(data)
    print(len(data))
    # return [(s, sents2users[s]) for s in data]
    return data


if __name__ == '__main__':
    data = preprocess_data(f'mydata/citations/citations.txt')
    f = open(f'mydata/citations/citations.tsv', 'w')
    for sentence in data:
        print(f'{sentence}', file=f)
    f.close()
