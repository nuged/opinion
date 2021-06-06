import pandas as pd
from data_preparation import tokenize, punctuation, stopwords
import pymorphy2
from multiprocessing import Pool

# df = pd.read_csv('mydata/labelled/results_20210514201313.tsv', quoting=3, index_col='id', sep='\t').dropna()
df = pd.read_csv('mydata/merged/opinion_scores.tsv', quoting=3, sep='\t', header=None)
morph = pymorphy2.MorphAnalyzer()
pd.set_option("display.max_rows", 100, "display.max_columns", None)

def extract(text, kwords):
    words = tokenize(text)
    words = [w for w in words if w.lower() not in stopwords and w.lower not in punctuation]
    results = []
    pos = 0
    for i, w in enumerate(words):
        pos = text.find(w, pos)
        lemmas = [p.normal_form for p in morph.parse(w)]
        for lemma in lemmas:
            if lemma in kwords:
                results.append((w, pos, pos + len(w)))
                break
        pos += len(w)
    return results

kwords = ["заговор", "ширма", "прикрытие", "происхождение", "чип", "чипировать",
                            "чипирование", "искусственный", "гейтс", "5G", "микросхема",
                            "бактериологический", "оружие", "война", "вживить", "вживлять", "вживление",
          "утечка", "лаборатория"]

with Pool(10) as p:
    extracted = p.starmap(extract, zip(df[1].values, [kwords] * df.shape[0]), chunksize=2048)

df['kwords'] = extracted
df = df[df['kwords'].astype(str) != '[]']

for i, row in df.iterrows():
    text = row[1]
    words = row.kwords
    for w, s, e in words:
        assert w == text[s: e]
# print(extract(df.text[11659], kwords))

df.to_csv('conspiracy.tsv', quoting=3, index=None, header=['user_id', 'text', 'p_opnion', 'kwords'], sep='\t')