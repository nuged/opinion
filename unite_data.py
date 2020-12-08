from unicodedata import normalize


def unite(cls, outfile):
    collector = []
    if cls == 'pos':
        file1 = 'data/positive.txt'
    else:
        file1 = 'data/negative.txt'
    file2 = f'data/{cls}_chosen.txt'
    with open(file1) as f:
        for line in f:
            line = line.strip()
            line = normalize("NFKC", line)
            if len(line.split('\t')) == 3:
                expression = line.split('\t', 1)[0]
                collector.append(expression)
            else:
                collector.append(line)
    with open(file2) as f:
        for line in f:
            line = line.strip()
            line = normalize("NFKC", line)
            collector.append(line)

    collector = list(filter(lambda x: x, collector))

    with open(outfile, 'w') as f:
        f.write("\n".join(collector))


if __name__ == "__main__":
    unite('pos', "data/pos.txt")
    unite('neg', 'data/neg.txt')
