from data_preparation import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_tokens(text):
    text = tokenize(text)
    text = map(lemmatize, text)
    text = filter(lambda x: x not in stopwords and x not in punctuation, text)
    return list(text)


def get_similar(texts):
    p = Pool(6)
    texts = p.map(get_tokens, texts)
    p.close()

    vec = TfidfVectorizer(lowercase=False, tokenizer=lambda x: x, preprocessor=lambda x: x)
    X = vec.fit_transform(texts)

    idxs = {}

    for i in range(X.shape[0] - 1):
        if i % 1000 == 0:
            print('proccessing {}-th step'.format(i))
        similarities = cosine_similarity(X[i], X[i + 1:])
        idx = []
        for j, sim in enumerate(similarities[0]):
            if sim > 0.6:
                idx.append(j)
        if idx:
            idxs[i] = idx

    for i in idxs:
        idxs[i] = list(map(lambda x: x + i + 1, idxs[i]))  # X is similar to X + 1 + Y

    c = 7
    for i in idxs:
        if c == 7:
            break
        main = texts[i]
        others = [texts[j] for j in idxs[i]]
        print('main:\t', main)
        print('similar sents:')
        for s in others:
            print(s)
        c += 1

    return idxs


def get_longest(idxs, texts):
    consider = [texts[i] for i in idxs]
    line = max(consider, key=lambda x: len(x))
    return line


def remove_similar(idxs, texts):
    written = set()
    result = []
    for i, line in enumerate(texts):
        if i not in idxs and i not in written:
            result.append(line)
        elif i not in written:
            result.append(get_longest([i] + idxs[i], texts))
            written.update([i] + idxs[i])
    return result


def write(file, data):
    with open(file, 'w') as f:
        f.write("\n".join(data))


if __name__ == '__main__':
    for p in ['pos', 'neg']:
        texts = read_data(f'mydata/{p}_processed.txt')
        ids = get_similar(texts)
        print("БЫЛО", len(texts))
        texts = remove_similar(ids, texts)
        print("СТАЛО", len(texts))
        write(f'{p}_final.txt', texts)


