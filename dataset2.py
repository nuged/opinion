import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 100, "display.max_columns", None)


def make_relevance_nli(filename):
    mapping = {'Релевантно маскам': '', "Релевантно вакцинированию": '_vaccines',
               "Релевантно карантину": '_quarantine', "Релевантно правительству": '_government'}

    df = pd.read_csv(f'mydata/labelled/masks/masks_rel_train.tsv', index_col=['text_id'], quoting=3, sep='\t')
    for t in ['government', 'vaccines', 'quarantine']:
        tmp = pd.read_csv(f'mydata/labelled/{t}/{t}_rel_train.tsv', index_col=['text_id'], quoting=3, sep='\t')
        df = df.join(tmp['relevant'], rsuffix=f'_{t}')

    data = pd.merge(df, pd.Series(mapping.keys(), name='hypothesis'), how='cross')

    data['label'] = data.apply(lambda row: row['relevant' + mapping[row.hypothesis]], axis=1)
    data = data[['text', 'hypothesis', 'label']]

    data.to_csv(filename + '_train.tsv', sep='\t', quoting=3, index=False)

    df = pd.read_csv(f'mydata/labelled/masks/masks_rel_test.tsv', index_col=['text_id'], quoting=3, sep='\t')
    for t in ['government', 'vaccines', 'quarantine']:
        tmp = pd.read_csv(f'mydata/labelled/{t}/{t}_rel_test.tsv', index_col=['text_id'], quoting=3, sep='\t')
        df = df.join(tmp['relevant'], rsuffix=f'_{t}')

    data = pd.merge(df, pd.Series(mapping.keys(), name='hypothesis'), how='cross')

    data['label'] = data.apply(lambda row: row['relevant' + mapping[row.hypothesis]], axis=1)
    data = data[['text', 'hypothesis', 'label']]

    data.to_csv(filename + '_test.tsv', sep='\t', quoting=3, index=False)


def make_sentiment_nli(filename):
    mapping = {'маскам': '', "вакцинированию": '_vaccines', "карантину": '_quarantine', "правительству": '_government'}

    def foo(row):
        s, t = row.hypothesis.split()[::2]
        if np.isnan(row['sen' + mapping[t]]):
            return np.nan
        return row['sen' + mapping[t]] == value[s]

    df = pd.read_csv(f'mydata/labelled/masks/masks_sen_train.tsv', index_col=['text_id'], quoting=3, sep='\t')[
        ['sen']]
    for t in ['government', 'vaccines', 'quarantine']:
        tmp = pd.read_csv(f'mydata/labelled/{t}/{t}_sen_train.tsv', index_col=['text_id'], quoting=3, sep='\t')
        df = df.join(tmp['sen'], rsuffix=f'_{t}', how='outer')

    total = pd.read_csv(f'mydata/labelled/masks/masks_rel_train.tsv', index_col=['text_id'], quoting=3, sep='\t')
    df = df.join(total[['text']])
    df = pd.merge(df, pd.Series([f'{sent} к {theme}' for sent in ['Позитивно', "Негативно", "Нейтрально"]
                                 for theme in ['маскам', "вакцинированию", "карантину", "правительству"]],
                                name='hypothesis'), how='cross')
    value = {'Позитивно': 2, "Негативно": 0, "Нейтрально": 1}

    df['label'] = df.apply(foo, axis=1)
    df = df[['text', 'hypothesis', 'label']].dropna()
    df['label'] = df.label.astype(int)

    df.to_csv(filename + '_train.tsv', sep='\t', quoting=3, index=False)

    df = pd.read_csv(f'mydata/labelled/masks/masks_sen_test.tsv', index_col=['text_id'], quoting=3, sep='\t')[
        ['sen']]
    for t in ['government', 'vaccines', 'quarantine']:
        tmp = pd.read_csv(f'mydata/labelled/{t}/{t}_sen_test.tsv', index_col=['text_id'], quoting=3, sep='\t')
        df = df.join(tmp['sen'], rsuffix=f'_{t}', how='outer')

    total = pd.read_csv(f'mydata/labelled/masks/masks_rel_test.tsv', index_col=['text_id'], quoting=3, sep='\t')
    df = df.join(total[['text']])
    df = pd.merge(df, pd.Series([f'{sent} к {theme}' for sent in ['Позитивно', "Негативно", "Нейтрально"]
                                 for theme in ['маскам', "вакцинированию", "карантину", "правительству"]],
                                name='hypothesis'), how='cross')
    value = {'Позитивно': 2, "Негативно": 0, "Нейтрально": 1}

    df['label'] = df.apply(foo, axis=1)
    df = df[['text', 'hypothesis', 'label']].dropna()
    df['label'] = df.label.astype(int)

    df.to_csv(filename + '_test.tsv', sep='\t', quoting=3, index=False)


def make_sentiment_qa(filename):
    mapping = {'Маски': '', "Вакцинирование": '_vaccines',
               "Карантин": '_quarantine', "Правительство": '_government'}

    df = pd.read_csv(f'mydata/labelled/masks/masks_sen_train.tsv', index_col=['text_id'], quoting=3, sep='\t')[
        ['sen']]
    for t in ['government', 'vaccines', 'quarantine']:
        tmp = pd.read_csv(f'mydata/labelled/{t}/{t}_sen_train.tsv', index_col=['text_id'], quoting=3, sep='\t')
        df = df.join(tmp['sen'], rsuffix=f'_{t}', how='outer')

    total = pd.read_csv(f'mydata/labelled/masks/masks_rel_train.tsv', index_col=['text_id'], quoting=3, sep='\t')
    df = df.join(total[['text']])

    data = pd.merge(df, pd.Series(mapping.keys(), name='question'), how='cross')
    data['label'] = data.apply(lambda row: row['sen' + mapping[row.question]], axis=1)
    data = data[['text', 'question', 'label']]
    data.dropna(inplace=True)
    data.label = data.label.astype(int)
    data.to_csv(filename + '_train.tsv', sep='\t', quoting=3, index=False)

    df = pd.read_csv(f'mydata/labelled/masks/masks_sen_test.tsv', index_col=['text_id'], quoting=3, sep='\t')[
        ['sen']]
    for t in ['government', 'vaccines', 'quarantine']:
        tmp = pd.read_csv(f'mydata/labelled/{t}/{t}_sen_test.tsv', index_col=['text_id'], quoting=3, sep='\t')
        df = df.join(tmp['sen'], rsuffix=f'_{t}', how='outer')

    total = pd.read_csv(f'mydata/labelled/masks/masks_rel_test.tsv', index_col=['text_id'], quoting=3, sep='\t')
    df = df.join(total[['text']])

    data = pd.merge(df, pd.Series(mapping.keys(), name='question'), how='cross')
    data['label'] = data.apply(lambda row: row['sen' + mapping[row.question]], axis=1)
    data = data[['text', 'question', 'label']]
    data.dropna(inplace=True)
    data.label = data.label.astype(int)
    data.to_csv(filename + '_test.tsv', sep='\t', quoting=3, index=False)


if __name__ == '__main__':
    make_relevance_nli('mydata/labelled/relevance_nli')
    make_sentiment_qa('mydata/labelled/sentiment_qa')
    make_sentiment_nli('mydata/labelled/sentiment_nli')
