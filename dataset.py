import pandas as pd

values = {'негативное': 'neg', 'релевантно вопросу': 'rel', 'не релевантно вопросу': 'irrel', 'нет ответа': 'NA',
              'позитивное': 'pos', 'релевантно, но невозможно определить': 'impos', 'нейтральное': 'neut',
              'и позитивное и негативное': 'posneg'}


def read_file(filename):
    df = pd.read_csv(filename, index_col=['id'], quoting=3, sep='\t').dropna()
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
    counts = df.pivot_table(index='text', columns=theme, aggfunc='size', fill_value=0)
    # counts = counts.apply(lambda x: x / x.sum(), axis=1)
    texts = df[['text_id', 'text', 'p_opinion']]
    texts = texts[~texts.text.duplicated()].set_index('text_id')
    res = texts.join(counts, on='text')
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
        comments = df.groupby('text').filter(lambda x: (x[theme] != 'irrel').any())
    else:
        comments = df
        theme = 'Total'
    comments = comments.groupby('text')['user'].count()
    comments = comments.value_counts().rename(theme)
    return comments


def write_statistics(df):
    counts = pd.DataFrame(columns=[1, 2, 3, 4, 5, 6])
    for t in [None, 'masks', 'vaccines', 'quarantine', 'government']:
        cdf = count_annotators(df, t)
        counts = counts.append(cdf)
    counts['number'] = counts.sum(axis=1).astype(int)
    counts.to_csv('mydata/labelled/annotator_counts.tsv', sep='\t')

    stat = pd.DataFrame(columns=[1, 2, 3, 4, 5, 6])
    for t in ['masks', 'vaccines', 'quarantine', 'government']:
        cdf = count_by_theme(df, t)
        stat = stat.append(statistics(cdf, t))
    stat.to_csv('mydata/labelled/statistics.tsv', sep='\t')


def score_opinion(df):
    df['p_opinion'] = 0
    return
    from bertcls import Klassifier, device, myDataset, predict
    import torch
    from torch.utils.data import DataLoader
    import numpy as np

    model = Klassifier().to(device)
    checkpoint = torch.load(f'models/opinion_cls_final.pt', map_location=device)
    model.load_state_dict(checkpoint['model'])

    data = df.text.tolist()
    labels = [1] * len(data)
    ds = myDataset(range(len(data)), data, labels)
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    _, _, y_probs = predict(model, dl)
    df['p_opinion'] = np.array(y_probs)[:, 1]


if __name__ == '__main__':
    pd.set_option("display.max_rows", 100, "display.max_columns", None)

    filename = 'mydata/labelled/results_20210619121105.tsv'
    df = read_file(filename)
    # df['p_opinion'] = 0
    unique_texts = df.drop_duplicates(subset='text')

    score_opinion(unique_texts)

    unique_texts = unique_texts[['text', 'p_opinion']]

    df = pd.merge(df, unique_texts, on='text')

    write_statistics(df)

    cols = list(values.values())
    cols.remove('NA')
    cols.remove('rel')

    f = open('mydata/labelled/dataset_stats.tsv', 'w')
    print("theme\tn_relevant\tn_positive\tn_negative\tn_other", file=f)
    for t in ['masks', 'vaccines', 'quarantine', 'government']:
        counts = count_by_theme(df, t)

        relevant = counts[(counts.irrel == 0) | (counts.rel > 1)]
        positive = relevant[relevant.pos > 0]
        negative = relevant[relevant.neg > 0]

        intersection_ids = positive.index.intersection(negative.index)
        intersection = positive.loc[intersection_ids]
        print(t, len(intersection_ids))

        intersection.to_csv(f'mydata/labelled/{t}/{t}_overlap.tsv', sep='\t', quoting=3)

        positive = positive.drop(intersection_ids)
        negative = negative.drop(intersection_ids)

        pos_ids = intersection.pos > intersection.neg
        positive = positive.append(intersection[pos_ids])
        neg_ids = intersection.neg > intersection.pos
        negative = negative.append(intersection[neg_ids])

        other = relevant.drop(positive.index.union(negative.index),)

        print(f"{t}\t{relevant.shape[0]}\t{positive.shape[0]}\t{negative.shape[0]}\t{other.shape[0]}", file=f)

        counts['relevant'] = 0
        counts.loc[relevant.index, 'relevant'] = 1
        counts = counts[['text', 'relevant', 'p_opinion']]
        counts.to_csv(f'mydata/labelled/{t}/{t}_relevance.tsv', sep='\t', quoting=3)

        # relevant = relevant.drop(intersection_ids)
        # relevant = relevant.drop(columns=cols)
        relevant = relevant[['text', 'p_opinion']]
        relevant['sen'] = 0
        relevant.loc[positive.index, 'sen'] = 2
        relevant.loc[other.index, 'sen'] = 1
        relevant.to_csv(f'mydata/labelled/{t}/{t}_sentiment.tsv', sep='\t', quoting=3)
    f.close()
