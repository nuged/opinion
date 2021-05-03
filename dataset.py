import pandas as pd

values = {'негативное': 'neg', 'релевантно вопросу': 'rel', 'не релевантно вопросу': 'irrel', 'нет ответа': 'NA',
              'позитивное': 'pos', 'релевантно, но невозможно определить': 'impos', 'нейтральное': 'neut',
              'и позитивное и негативное': 'posneg'}


def read_file(filename):
    df = pd.read_csv(filename, index_col=['id'], quoting=3, sep='\t')
    for t in ['masks', 'quarantine', 'vaccines', 'government']:
        df[t] = df[t].apply(lambda x: values[x.split(' - ')[1]])
    # print((df == 'NA').any(axis=1))
    df = df[~(df == 'NA').any(axis=1)]
    return df


def count_by_theme(df, theme):
    """
    Counts values for the theme for each text in DataFrame
    :param df: pd.DataFrame
    :param theme: string
    :return: pd.DataFrame with unique texts and their counts
    """
    counts = df.pivot_table(index='text_id', columns=theme, aggfunc='size', fill_value=0)
    # counts = counts.apply(lambda x: x / x.sum(), axis=1)
    texts = df[['text_id', 'text']].drop_duplicates().set_index('text_id')
    res = texts.join(counts, on='text_id')
    res['rel'] = res[['impos',  'irrel',  'neg',  'neut',  'pos',  'posneg']].sum(axis=1)\
                 - res['irrel']
    return res


def statistics(count_df, theme=None):
    df = pd.DataFrame(columns=[1, 2, 3])
    for name, short in values.items():
        if short == 'NA':
            continue
        if theme is not None:
            name = theme + "__" + name
        counts = count_df.groupby(short)['text'].nunique().rename(name)
        counts = counts[::-1]  # reversing series to do right-to-left cumsum
        counts = counts.cumsum().loc[:1]  # keep indexes from 3 to 1
        df = df.append(counts)
    df = df.fillna(0).astype(int)
    return df


def count_annotators(df, theme=None):
    if theme is not None:
        comments = df.groupby('text_id').filter(lambda x: (x[theme] != 'irrel').any())
    else:
        comments = df
        theme = 'Total'
    comments = comments.groupby('text_id')['user'].count()
    comments = comments.value_counts().rename(theme)
    return comments


def write_statistics(df):
    counts = pd.DataFrame(columns=[1, 2, 3, 4])
    for t in [None, 'masks', 'vaccines', 'quarantine', 'government']:
        cdf = count_annotators(df, t)
        counts = counts.append(cdf)
    counts['number'] = counts.sum(axis=1).astype(int)
    counts.to_csv('mydata/labelled/annotator_counts.tsv', sep='\t')

    stat = pd.DataFrame(columns=[1, 2, 3, 4])
    for t in ['masks', 'vaccines', 'quarantine', 'government']:
        cdf = count_by_theme(df, t)
        stat = stat.append(statistics(cdf, t))
    stat.to_csv('mydata/labelled/statistics.tsv', sep='\t')


if __name__ == '__main__':
    pd.set_option("display.max_rows", 100, "display.max_columns", None)

    filename = 'mydata/labelled/results_20210430115452.tsv'
    df = read_file(filename)

    write_statistics(df)

    cols = list(values.values())
    cols.remove('NA')
    cols.remove('rel')

    f = open('mydata/labelled/dataset_stats.tsv', 'w')
    print("theme\tn_relevant\tn_positive\tn_negative\tn_other", file=f)
    for t in ['masks', 'vaccines', 'quarantine', 'government']:
        counts = count_by_theme(df, t)

        relevant = counts[counts.irrel < counts[cols].sum(axis=1)]
        positive = counts[counts.pos > 0]
        negative = counts[counts.neg > 0]

        intersection_ids = positive.index.intersection(negative.index)
        intersection = positive.loc[intersection_ids]
        print(t, len(intersection_ids))

        positive = positive.drop(intersection_ids)
        negative = negative.drop(intersection_ids)

        pos_ids = intersection.pos > intersection.neg
        positive = positive.append(intersection[pos_ids])
        neg_ids = intersection.neg > intersection.pos
        negative = negative.append(intersection[neg_ids])

        other = relevant.drop(positive.index.union(negative.index))

        print(f"{t}\t{relevant.shape[0]}\t{positive.shape[0]}\t{negative.shape[0]}\t{other.shape[0]}", file=f)

        counts['relevant'] = 0
        counts.loc[relevant.index, 'relevant'] = 1
        counts = counts[['text', 'relevant']]
        counts.to_csv(f'mydata/labelled/{t}/{t}_relevance.tsv', sep='\t')

        # relevant = relevant.drop(intersection_ids)
        # relevant = relevant.drop(columns=cols)
        relevant = relevant[['text']]
        relevant['sentiment'] = 0
        relevant.loc[positive.index, 'sentiment'] = 2
        relevant.loc[other.index, 'sentiment'] = 1
        relevant.to_csv(f'mydata/labelled/{t}/{t}_sentiment.tsv', sep='\t')
    f.close()
