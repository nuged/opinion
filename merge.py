from ria_themes import themes
import pandas as pd

pd.set_option("display.max_rows", None, "display.max_columns", None)


for t in themes:
    a = 'mydata/RIA/opinion classified/ria_scores_by_user.tsv'
    b = 'mydata/65_sources/opinion_scores.tsv'
    a = pd.read_csv(a, index_col=False, quoting=3, sep='\t', header=None).drop(columns=[3])
    b = pd.read_csv(b, index_col=False, quoting=3, sep='\t', header=None)
    print(f'merging dfs:\t{a.shape[0]} and {b.shape[0]}')
    result = pd.concat([a, b], ignore_index=True)
    result = result.sort_values([0, 2], axis=0, ascending=[True, False])
    print(result.shape[0])
    result.drop_duplicates(subset=[1], inplace=True)
    print(result.shape[0])
    result.to_csv(f'mydata/merged/opinion_scores.tsv', sep='\t', index=False, quoting=3, header=None)
    break

