for file in ['data/neg.txt', 'data/pos.txt']:
    with open(file) as f, open(file + ".corr") as g:
        for lineA, lineB in zip(f, g):
            if lineA != lineB:
                print(lineA.strip())
                print(lineB.strip())
                print()
