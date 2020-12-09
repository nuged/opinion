import pymorphy2
import re
from multiprocessing import Pool

morph = pymorphy2.MorphAnalyzer()


def lower(match):
    with open('opinion/alphabet_rus.txt', encoding='cp1251') as f:
        wlist = set(f.read().strip().split('\n'))

    word = match.group(0)
    if morph.parse(word.lower())[0].normal_form in wlist:
        return word.lower()
    else:
        return word


def remove_caps(text):
    capitalize = lambda m: m.group(0).capitalize()
    text = re.sub(r'\b[А-ЯЁ]+\b', lower, text)
    text = re.sub(r'^[а-яё]+\b', capitalize, text)
    return text


def process_data(data, num_processes=12, chunk_size=64):
    p = Pool(num_processes)
    res = p.map(remove_caps, data, chunksize=chunk_size)
    p.close()
    return res


if __name__ == "__main__":
    for file in ['data/pos.txt', 'data/neg.txt']:
        with open(file) as f:
            data = f.read().strip().split("\n")
        texts = process_data(data)
        with open(file + ".corr", "w") as f:
            f.write("\n".join(texts))
