import os
import re
import pymorphy2
from nltk.tokenize import sent_tokenize, word_tokenize
from multiprocessing import Pool


morph = pymorphy2.MorphAnalyzer()
wordset = {w.strip() for w in open('opinion/alphabet_rus.txt', encoding='cp1251')}

remove_digits = True


def process_text(text):
    sents = []
    for subt in text.split('\n'):
        if subt:
            s = sent_tokenize(subt, language='russian')
            sents.extend(s)

    found = []
    empty = []
    for s in sents:
        s = s.strip()
        # r'(^|[\s(])[\d,+.-]+($|[%.!?)\s])'
        words = word_tokenize(s, language='russian')
        if len([w for w in words if w.isalnum()]) < 5:
            continue
        if remove_digits and re.search(r'(^|[^\w\-:.])[+-]?\d+($|[^\w\-\:])', s):
            empty.append(s)
            continue
        flag = False
        for word in words:
            if not word.isalnum():
                continue
            if morph.parse(word)[0].normal_form in wordset:
                flag = True
                s = re.sub(r'\b(' + word + r')\b', word.upper(), s)
        if flag:
            found.append(s)
        else:
            empty.append(s)
    return found, empty


def read_data(directory):
    pattern = re.compile('<div class=\"text-block\">([\s\S]+)</div>')
    texts = []
    for d in os.listdir(directory):
        for file in os.listdir(os.path.join('data', d)):
            path = os.path.join('data', d, file)
            with open(path) as f:
                data = f.read()
            text = pattern.search(data).group(1).strip()
            texts.append(text)
    return texts


data = read_data('data')

p = Pool(processes=4)
results = p.map(process_text, data)

found = []
empty = []
for i, r in enumerate(results):
    if r is None:
        continue
    found.extend(r[0])
    empty.extend(r[1])


with open(f'found{"_nonums" if remove_digits else ""}.txt', 'w') as f:
    f.write('\n'.join(found))

with open(f'rest{"_nonums" if remove_digits else ""}.txt', 'w') as f:
    f.write('\n'.join(empty))
