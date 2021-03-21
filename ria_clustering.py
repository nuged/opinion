"""
Performs clustering of sentences by creating their embeddings
"""

from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import pandas as pd
import pickle
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from data_preparation import stopwords, punctuation, tokenize, lemmatize
from sklearn.metrics.pairwise import cosine_similarity
import os


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased-conversational")

    def forward(self, *args, **kwargs):
        x = self.bert(*args, **kwargs)
        return x.pooler_output


def encode(theme):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(10)

    encoder = Encoder().to(device)
    encoder.eval()
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-conversational')
    tokenizer.add_special_tokens({'additional_special_tokens': ['[USER]']})

    data = pd.read_csv(f'mydata/ria_{theme}_scores.tsv', index_col=False, quoting=3, sep='\t')
    vectors = []
    with torch.no_grad():
        for idx, row in data.iterrows():
            if idx % 50 == 49:
                print(idx)
            input = tokenizer(row['text'], padding=True, return_tensors='pt')
            output = encoder(**input).detach().cpu().numpy().reshape(768)
            vectors.append(output)
    with open(f'mydata/RIA/vectors/{theme}.pk', 'wb') as f:
        pickle.dump(vectors, f)
    return vectors


def load_embeddings(theme):
    with open(f'mydata/RIA/vectors/{theme}.pk', 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def clusterize(embeddings):
    umap_embeddings = umap.UMAP(n_neighbors=15,
                                n_components=10,
                                metric='cosine', random_state=7).fit_transform(embeddings)

    # import hdbscan
    # cluster = hdbscan.HDBSCAN(min_cluster_size=2,
    #                           metric='euclidean',
    #                           cluster_selection_method='eom', approx_min_span_tree=False).fit(umap_embeddings)

    km = KMeans(n_clusters=3, max_iter=1500, n_init=10, random_state=7)
    cluster = km.fit(umap_embeddings)
    return cluster


def plot_clusters(labels, umap_data, theme):
    # Prepare data

    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = labels

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(18, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=2)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=2, cmap='cool')
    plt.colorbar()
    plt.title(theme)
    # plt.savefig(f'plots/{theme}.png')
    plt.show()


def get_docs(theme, cluster):
    """
    Writes cluster_ids to the dataframes, also gets merged documents for the clusters
    :param theme:
    :param cluster:
    :return:
    """
    docs_df = pd.read_csv(f'mydata/RIA/topics/scored data/ria_{theme}_scores.tsv',
                          index_col=False, quoting=3, sep='\t')
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(docs_df.shape[0])
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'text': ' '.join})
    return docs_df, docs_per_topic


def c_tf_idf(documents):
    """
    Calculates tfidf vectors for the documents and returns them and the vectorizer
    :param documents:
    :return:
    """
    documents = list(map(tokenize, documents))
    documents = list(map(lambda x: list(map(lemmatize, x)), documents))
    documents = [list(filter(lambda x: x not in stopwords and x not in punctuation, doc)) for doc in documents]
    count = TfidfVectorizer(lowercase=False, tokenizer=lambda x: x, preprocessor=lambda x: x).fit(documents)
    tf_idf = count.transform(documents).toarray()
    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    indices = tf_idf.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf[i][j])
                           for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .text
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "text": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes


def get_similarities(docs_df, tf_idf, count):
    similarities = [[], [], []]
    for doc in docs_df['text']:
        doc = tokenize(doc)
        doc = list(map(lemmatize, doc))
        doc = list(filter(lambda x: x not in stopwords and x not in punctuation, doc))
        vec = count.transform([' '.join(doc)])
        for t in [0, 1, 2]:
            sim = cosine_similarity(vec.reshape(1, -1), tf_idf[t].reshape(1, -1))
            similarities[t].append(sim[0][0])
    return similarities


def write_topwords(theme, top_n_words):
    f = open(f'mydata/{theme}_cluster_topwords.txt', 'w')
    for t in range(3):
        print(f'cluster #{t}', file=f)
        for word, score in top_n_words[t]:
            print(f"{word}\t{score:4.3f}", file=f)
        print(file=f)
    f.close()


if __name__ == '__main__':
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    df = pd.read_csv('mydata/RIA/topics/scored data/ria_vaccine_labelled.tsv', index_col=False, quoting=3, sep='\t')
    df.drop(columns=['sent_words'], inplace=True)
    labels = {'-': -1, '=': 0, '+': 1}
    labelled = list(map(lambda x: labels[x[0]], df['usr_id'].values))
    df['Topic'] = labelled

    # low = df.sentiment.min()
    # high = df.sentiment.max()
    # df.sentiment = 2 * (df.sentiment - low) / (high - low) - 1
    #
    # df.sentiment[df.absence == 1] *= -1
    #
    # plt.scatter(df.sentiment, df.Topic)
    # plt.show()
    #
    # # threshold = 0.05
    # # df.sentiment[df.sentiment < -threshold] = -1
    # # df.sentiment[(-threshold <= df.sentiment) & (df.sentiment <= threshold)] = 0
    # # df.sentiment[df.sentiment > threshold] = 1
    #
    # # print(df.drop(columns=['text']))
    #
    # from scipy.stats import pearsonr, spearmanr, kendalltau
    # print(pearsonr(df.sentiment, df.Topic))
    # print(spearmanr(df.sentiment, df.Topic))
    # print(kendalltau(df.sentiment, df.Topic))
    # exit(0)

    # embeddings = load_embeddings('vaccine')
    embeddings, _ = c_tf_idf(df.text.values)

    for metric in ['euclidean', 'chebyshev', 'correlation', 'minkowski', 'wminkowski', 'seuclidean']:
        umap_data = umap.UMAP(n_neighbors=15, n_components=2,
                              metric=metric, random_state=7).fit_transform(embeddings)
        plot_clusters(df['Topic'], umap_data, f'test_{metric}')


    exit(0)
    for theme in ['covid', 'vaccine', 'masks', 'lockdown']:
        if os.path.exists(f'mydata/RIA/vectors/{theme}.pk'):
            print('loading embeddings from disk')
            embeddings = load_embeddings(theme)
        else:
            print('creating embeddings')
            embeddings = encode(theme)

        clustering = clusterize(embeddings)
        plot_clusters(clustering.labels_, embeddings, theme)
        docs_df, _ = get_docs(theme, clustering)
        print(docs_df)
        # tf_idf, count = c_tf_idf(docs_per_topic.text.values)
        #
        # print(tf_idf.shape, tf_idf.min(), tf_idf.max())
        #
        # top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=50)
        # topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)
        #
        # write_topwords(theme, top_n_words)
        #
        # print(topic_sizes)
        #
        # similarities = get_similarities(docs_df, tf_idf, count)
        #
        # for t in [0, 1, 2]:
        #     docs_df[f'sim_{t}'] = similarities[t]
        #
        # docs_df.drop(['Doc_ID'], axis=1, inplace=True)
        # print(docs_df)
        # docs_df.to_csv(f'mydata/{theme}_cluster.tsv', sep='\t', index=False, quoting=3)
        #
        # for t in [0, 1, 2]:
        #     tmp = docs_df.sort_values(by=f'sim_{t}', ascending=False)
        #     tmp = tmp[tmp.Topic == t]
        #     topdocs = tmp.iloc[:50][['text', f'sim_{t}']]
        #     topdocs.to_csv(f'mydata/{theme}_cluster_topdocs_{t}.tsv',  sep='\t', index=False, quoting=3)
