import pandas as pd

# data = pd.read_csv("pos_classified.txt", sep='\t', header=None, quoting=3)
# print((data))
# data.drop_duplicates(inplace=True)
# print((data))
# data.sort_values(by=2, axis=0, ascending=True, inplace=True)
#
# data.to_csv('pos_classified.csv', sep='\t', header=False, index=False)

df = pd.read_csv('mydata/RIA/topics/clustering/covid_cluster.tsv', sep='\t', quoting=3)

for t in [0, 1, 2]:
    tmp = df.sort_values(by=f'sim_{t}', ascending=False)
    tmp = tmp[tmp.Topic == t]
    topdocs = tmp.iloc[:50][['text', f'sim_{t}']]
    topdocs.to_csv(f'mydata/covid_cluster_topdocs_{t}.tsv',  sep='\t', index=False, quoting=3)
