from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from string import punctuation
from multiprocessing import Pool
from collections import defaultdict

tok = TweetTokenizer()
stop_words = set(stopwords.words('english') + ['’'])
stop_words |= set(map(lambda w: w.capitalize(), stop_words))
punctuation += '“”'
punctuation = list(punctuation) + ['...']
punctuation.remove('*')
punctuation.remove('/')
punctuation.remove('\\')
punctuation.remove('&')

NGRAMS = 2


def split_punkt(words):
    res = []
    sub_words = []
    for w in words:
        if w in punctuation:
            res.append(sub_words)
            sub_words = []
        else:
            sub_words.append(w)
    res.append(sub_words)
    return res


def replace_all(word):
    return word.replace("'", '').replace('#', '').replace('-', '')


def get_ngrams(text, n=NGRAMS):
    # text = preprocess(text.strip())
    words = tok.tokenize(text)
    if not words:
        return None
    if words[-1] == '…':
        words = words[:-2]
    words = filter(lambda word:
                   word not in stop_words and replace_all(word).isalnum()
                   and not word.isdigit(),
                   words)
    words = list(map(str.lower, words))
    words = list(map(lambda x: x[1:] if x.startswith('#') else x, words))
    words = split_punkt(words)
    res = []
    for sent in words:
        ngrams = zip(*[sent[i:] for i in range(n)])
        res.extend(ngrams)
    return res


def count_ngrams(filename, num_threads=4):
    pool = Pool(num_threads)
    f = open(filename)
    result = pool.imap(get_ngrams, f, chunksize=512)
    ngrams_count = defaultdict(int)
    for r in result:
        if r is None:
            continue
        for ngram in r:
            ngrams_count[ngram] += 1
    f.close()
    pool.close()
    return ngrams_count


def write_counts(counts, filename):
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    with open(filename, 'w') as f:
        for ngram, num in counts:
            print(ngram, '\t', num, file=f)


def get_wc(filename):
    with open(filename) as f:
        wc = map(lambda x: x.split('\t'), f.read().strip().split('\n'))
        wc = {x[0]: int(x[1]) for x in wc if int(x[1]) >= 10}
    return wc


def weirdness(norm_wc, cov_wc, min_freq=10):
    common = {}
    for w in norm_wc:
        common[w] = [norm_wc[w], min_freq]
    for w in cov_wc:
        if w in common:
            common[w][1] = cov_wc[w]
        else:
            common[w] = [min_freq, cov_wc[w]]
    num_norm = 0
    num_cov = 0
    for c_norm, c_cov in common.values():
        num_norm += c_norm
        num_cov += c_cov
    scores = {w: (common[w][1] / num_cov) / (common[w][0] / num_norm) for w in common}
    return scores


def write_results(filename, scores):
    with open(filename, 'w') as f:
        for word, score in sorted(scores.items(), key=lambda x: -x[1]):
            f.write(f'{word}\t{score}\n')


if __name__ == '__main__':
    for n_words in [1, 2]:
        print(f'N_WORDS {n_words}!')
        path = f'data/selected/selected_{n_words}.txt'
        counts = count_ngrams(path, num_threads=4)
        write_counts(counts, f'Results/ngrams_counts/selected_{n_words}_N{NGRAMS}.txt')

    print("general")
    path = '/home/nuged/general.txt'
    counts = count_ngrams(path, num_threads=4)
    write_counts(counts, f'Results/ngrams_counts/general_N{NGRAMS}.txt')

    for NGRAMS in [1, 2]:
        base_wc = get_wc(f'Results/ngrams_counts/general_N{NGRAMS}.txt')
        for NWORDS in [1, 2]:
            filename = f'Results/ngrams_counts/selected_{NWORDS}_N{NGRAMS}.txt'
            wc = get_wc(filename)
            weird = weirdness(base_wc, wc)
            write_results(f'Results/Scores/weird_sel_{NWORDS}_N{NGRAMS}.txt', weird)