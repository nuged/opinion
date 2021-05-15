import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from data_preparation import lemmatize, tokenize, stopwords, punctuation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix
from sklearn.dummy import DummyClassifier
import pandas as pd

TASK = 'relevance'


def prepare_data(texts):
    data = texts
    data = list(map(tokenize, data))
    data = list(map(lambda x: list(map(lemmatize, x)), data))
    data = list(map(lambda x: list(filter(lambda y: y not in punctuation and y not in stopwords, x)), data))
    return data


def cross_validation(model, data, labels, n_splits=4):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3)
    predictions = []
    gts = []
    for train_idx, test_idx in cv.split(data, labels):
        X_train, y_train = data[train_idx], labels[train_idx]
        X_test, y_test = data[test_idx], labels[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions.extend(y_pred)
        gts.extend(y_test)

    # if TASK == 'sentiment':
    #     c = confusion_matrix(gts, predictions)
    #     print(type(model).__name__, '\n', c / c.sum(axis=1))

    avg_mode = 'binary' if TASK == 'relevance' else 'macro'

    return {'accuracy': accuracy_score(gts, predictions),
            'precision': precision_score(gts, predictions, average=avg_mode),
            'recall': recall_score(gts, predictions, average=avg_mode),
            'f1': f1_score(gts, predictions, average=avg_mode)}


def write_topwords(svm_cls, data, labels, features, theme):
    svm_cls.fit(data, labels)
    coef = svm_cls.coef_.toarray()[0]
    with open(f"mydata/labelled/{TASK}/{theme}_topwords.txt", "w") as f:
        for idx in np.abs(coef).argsort()[-1::-1]:
            print(f"{features[idx]}\t{coef[idx]:3.2f}", file=f)


def write_results(results, theme, task=TASK):
    f = open(f'mydata/labelled/{task}/{theme}_comparison.tsv', 'a')
    print(f"name\taccuracy\tprecision\trecall\tf1", file=f)
    for model, scores in results.items():
        acc = scores['accuracy'] * 100
        pr = scores['precision'] * 100
        rec = scores['recall'] * 100
        f1 = scores['f1'] * 100
        print(f"{model}\t{acc:4.2f}\t{pr:4.2f}\t{rec:4.2f}\t{f1:4.2f}", file=f)
    f.close()


if __name__ == '__main__':
    for t in ['masks', 'quarantine', 'government', 'vaccines']:
        df = pd.read_csv(f'mydata/labelled/{t}/{t}_{TASK}.tsv', index_col=['text_id'], quoting=3, sep='\t')
        data = prepare_data(df.text.values)

        vec = TfidfVectorizer(lowercase=False, tokenizer=lambda x: x, preprocessor=lambda x: x, min_df=5,
                              ngram_range=(1, 1))

        data = vec.fit_transform(data)
        labels = df.sentiment.values if TASK == 'sentiment' else df.relevant.values

        svm_cls = SVC(kernel='linear', random_state=0, class_weight='balanced')
        multiNB_cls = MultinomialNB()
        bernNB_cls = BernoulliNB()
        rf_cls = RandomForestClassifier(random_state=0)
        gb_cls = GradientBoostingClassifier(random_state=0)
        dummy = DummyClassifier(random_state=0, strategy='most_frequent')
        results = {}
        for cls in [svm_cls, multiNB_cls, bernNB_cls, rf_cls, gb_cls, dummy]:
            model_name = type(cls).__name__
            scores = cross_validation(cls, data, labels, n_splits=5)
            results[model_name] = scores

        write_results(results, theme=t)
