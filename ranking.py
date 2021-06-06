import pandas as pd
from dataset import count_by_theme, read_file


if __name__ == "__main__":
    pd.set_option("display.max_rows", 100, "display.max_columns", None)
    filename = 'mydata/labelled/results_20210514201313.tsv'
    df = read_file(filename)

    for t in ['masks', 'vaccines', 'quarantine', 'government']:
        counts = count_by_theme(df, t)
        relevant = counts[counts.rel > 0]
        relevant['positive'] = relevant.pos
        relevant['negative'] = relevant.neg
        relevant['neutral'] = relevant.neut
        relevant['positive'] += relevant.posneg * 0.5
        relevant['negative'] += relevant.posneg * 0.5
        relevant['positive'] += relevant.impos * 0.33
        relevant['negative'] += relevant.impos * 0.33
        relevant['neutral'] += relevant.impos * 0.33
        relevant['total'] = relevant[['irrel', 'rel']].sum(axis=1)
        relevant = relevant[['text', 'positive', 'negative', 'neutral', 'total']]
        relevant.to_csv(f'mydata/labelled/ranking/{t}_rank.tsv', sep='\t', quoting=3)
