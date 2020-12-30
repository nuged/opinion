import pandas as pd
from data_preparation import *
from nltk.tokenize import sent_tokenize, TweetTokenizer
from multiprocessing import Pool
from random import sample


posts = pd.read_csv('data/ria/ria_posts.csv', index_col=None)
comments = pd.read_csv('data/ria/ria_comments.csv', index_col=None)
posts = posts.text.dropna().tolist()
comments = comments.text.dropna().tolist()


def norm(text):
    return normalize("NFKC", text)


def sent_tok(text):
    return sent_tokenize(text, language='russian')


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

data = [s for i, s in enumerate(data) if len(tokenized_data[i]) >= 5]
print(len(data))
data = set(data)
print(len(data))

f = open('mydata/ria_comments.txt', 'w')
for d in data:
    print(d, file=f)
f.close()
