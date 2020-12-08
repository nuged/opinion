import pandas as pd

data = pd.read_csv("pos_classified.txt", sep='\t', header=None, quoting=3)
print((data))
data.drop_duplicates(inplace=True)
print((data))
data.sort_values(by=2, axis=0, ascending=True, inplace=True)

data.to_csv('pos_classified.csv', sep='\t', header=False, index=False)
