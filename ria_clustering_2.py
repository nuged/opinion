"""
Sentiment clustering with lexical and syntactical features
"""

import pandas as pd
from ria_themes import process_line, themes
import numpy as np
from data_preparation import lemmatize, tokenize, stopwords, punctuation
from sklearn.metrics import f1_score, precision_score, recall_score
from ria_lexical_sentiment import sentiments


def get_sents_positions(words):
    sent_words = {}
    for (word, start, end) in words:
        negated = word.startswith(('не ', 'ни ', 'нет ', 'нету ')) and word not in sentiments
        if negated:
            sent = sentiments[word.split(' ', 1)[1]]
        else:
            sent = sentiments[word]
        sent_words[word] = [start, end, sent]
        if negated:
            sent_words[word][-1] *= -1
    return sent_words


def classify(row):
    if row.sent_words == '[]':
        if row.p_opinion <= 0.2:
            return 0
        else:
            return 999
    sent_words = get_sents_positions(eval(row.sent_words))
    keywords = eval(row.bounds)
    absences = eval(row.absence)
    results = []
    for i, (kw_start, kw_end) in enumerate(keywords):
        for word, (start, end, sent) in sent_words.items():
            if kw_start >= end:
                words_between = tokenize(row.text[end: kw_start])
            elif start >= kw_end:
                words_between = tokenize(row.text[kw_end: start])
            else:
                continue
            if any(x in words_between for x in [',', ';', '...']):
                continue
            words_between = map(lemmatize, words_between)
            words_between = list(filter(lambda x: x not in stopwords and x not in punctuation, words_between))
            if words_between:
                continue
            if absences[i]:
                results.append(-sent)
            else:
                results.append(sent)
    if results:
        score = sum(results)
        if score > 0:
            return 1
        if score < 0:
            return -1
        # return 0
    return 999


if __name__ == '__main__':
    for theme, theme_words in themes.items():
        print(f'processing {theme}')
        df = pd.read_csv(f'mydata/citations/{theme}_sentiment.tsv', index_col=False, quoting=3, sep='\t')
        df['lex_sentiment'] = df.apply(classify, axis=1)
        for sentiment, label in [('pos', 1), ('neg', -1), ('neut', 0)]:
            tmp = df[df.lex_sentiment == label]
            print(f'{sentiment}:\t{tmp.shape[0]} records')
            tmp.to_csv(f'mydata/citations/lexicon/{theme}_{sentiment}.tsv', sep='\t', header=True, index=False, quoting=3)
        print()
        df.to_csv(f'mydata/citations/lexicon/{theme}_total.tsv', sep='\t', header=True, index=False, quoting=3)
