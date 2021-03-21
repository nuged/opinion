import pandas as pd
from itertools import combinations


def combination_belongs_to_clustering(combination_ids, clustering):
    cluster = clustering[combination_ids[0]]
    for id in combination_ids:
        if clustering[id] != cluster:
            return False
    return True


df = pd.read_csv('mydata/clustering_test.tsv', index_col=None, sep='\t', quoting=3)
predicted = df[df.test != -1].test.tolist()
labelled = df[df.test != -1].Topic.tolist()

from sklearn.metrics import classification_report

print(classification_report(labelled, predicted))

# predicted = pd.read_csv(f'mydata/RIA/topics/clustering/vaccine_cluster.tsv',
#                         index_col=None, sep='\t', quoting=3)
# labelled = pd.read_csv(f'mydata/RIA/topics/scored data/ria_vaccine_labelled.tsv',
#                        index_col=None, sep='\t', quoting=3)

# labels = {'-': 0, '=': 1, '+': 2}
# labelled = list(map(lambda x: labels[x[0]], labelled['usr_id'].values))
# predicted = predicted.Topic.tolist()

print(predicted, labelled)

both = 0
pred = 0
lab = 0
none = 0
for comb in combinations(range(len(predicted)), 2):
    belongs_pred = combination_belongs_to_clustering(comb, predicted)
    belongs_lab = combination_belongs_to_clustering(comb, labelled)
    if belongs_pred and belongs_lab:
        both += 1
    elif belongs_pred:
        pred += 1
    elif belongs_lab:
        lab += 1
    else:
        none += 1

print(f'precision={100 * both / (both + pred):4.2f}')
print(f'recall={100 * both / (both + lab):4.2f}')
print(f'F1={100 * 2 * both / (2 * both + pred + lab):4.2f}')
print(f'Rand={100 * (both + none) / (both + none + pred + lab):4.2f}')
print(f'Jaccard={100 * both / (both + pred + lab):4.2f}')
