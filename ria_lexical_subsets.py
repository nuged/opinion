"""
Separates comments to opinion and feeling sets lexically
"""


from kwords import process_line
from multiprocessing import Pool

opinion = set()
feeling = set()

with open('opinion/rusentilex_2017.txt') as f:
    for line in f:
        if line.startswith('!'):
            continue
        data = line.strip().split(', ')
        if len(data) < 2:
            continue
        word = data[2]

        if '/' in word:
            words = word.split('/')
        else:
            words = [word]

        if 'feeling' in data:
            feeling.update(words)
        elif 'opinion' in data:
            opinion.update(words)

for week in [2, 3]:
    texts = []
    scores = []
    with open(f'mydata/ria_scores_{week}.tsv') as f:
        for line in f:
            line = line.strip()
            text, score, cls = line.split('\t')
            if cls == 'True':
                texts.append(text)
                scores.append(float(score))

    p = Pool(10)
    opinion_texts = p.starmap(process_line, [(t, opinion) for t in texts])
    feeling_texts = p.starmap(process_line, [(t, feeling) for t in texts])
    p.close()

    f = open(f'mydata/ria_opinion_{week}.tsv', 'w')
    for i, (text, words) in enumerate(opinion_texts):
        if not words:
            continue
        print(f'{text}\t{scores[i]}\t{words}', file=f)
    f.close()

    f = open(f'mydata/ria_feeling_{week}.tsv', 'w')
    for i, (text, words) in enumerate(feeling_texts):
        if not words:
            continue
        print(f'{text}\t{scores[i]}\t{words}', file=f)
    f.close()
