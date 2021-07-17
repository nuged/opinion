import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from data_preparation import lemmatize, tokenize, stopwords, punctuation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix
from sklearn.dummy import DummyClassifier
import pandas as pd
from tune_sklearn import TuneSearchCV
import ray
from multiprocessing import Pool
from scipy.sparse import hstack

ray.init(log_to_driver=False)

# TASK = 'sen'
stopwords.remove('не')
stopwords.remove('нет')
stopwords.remove("ни")


def tune_model(model, params, data, labels, TASK, rewrite=False):
    tune_search = TuneSearchCV(
        model,
        params,
        scoring='f1' if TASK == 'rel' else 'f1_macro',
        cv=5,
        search_optimization="bayesian",
        n_trials=8,
        random_state=0,
        verbose=0
    )
    tune_search.fit(data, labels)
    write_params(tune_search.best_estimator_, tune_search.best_params_, rewrite=rewrite)

    return tune_search.best_estimator_

def foo(spisok):
    return list(map(lemmatize, spisok))

def boo(spisok):
    return list(filter(lambda y: y not in punctuation and y not in stopwords, spisok))

def prepare_data(texts):
    data = texts
    with Pool(10) as p:
        data = p.map(tokenize, data, )
        data = p.map(foo, data)
        data = p.map(boo, data)
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

    # if TASK == 'sen':
    #     c = confusion_matrix(gts, predictions)
    #     print(type(model).__name__, '\n', c / c.sum(axis=1))

    avg_mode = 'binary' if TASK == 'rel' else 'macro'

    return {'accuracy': accuracy_score(gts, predictions),
            'precision': precision_score(gts, predictions, average=avg_mode),
            'recall': recall_score(gts, predictions, average=avg_mode),
            'f1': f1_score(gts, predictions, average=avg_mode)}


def test(model, data, labels):
    predictions = model.predict(data)
    gts = labels
    avg_mode = 'binary' if TASK == 'rel' else 'macro'
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


def write_results(results, theme, TASK):
    f = open(f'mydata/labelled/{TASK}/{theme}_comparison.tsv', 'a')
    print(f"name\taccuracy\tprecision\trecall\tf1", file=f)
    for model, scores in results.items():
        acc = scores['accuracy'] * 100
        pr = scores['precision'] * 100
        rec = scores['recall'] * 100
        f1 = scores['f1'] * 100
        print(f"{model} & {acc:4.2f} & {pr:4.2f} & {rec:4.2f} & {f1:4.2f}", file=f)
    f.close()


def write_params(model, params, rewrite=False):
    with open(f'mydata/labelled/{TASK}/params.txt', 'w' if rewrite else 'a') as f:
        print(f'({t}, {type(model).__name__}, {params})', file=f)


if __name__ == '__main__':
    for t in ['masks', 'quarantine', 'government', 'vaccines']:
        for TASK in ['sen', 'rel']:
            print(t, TASK)
            df = pd.read_csv(f'mydata/labelled/{t}/{t}_{TASK}_train.tsv', index_col=['text_id'], quoting=3, sep='\t')
            data = prepare_data(df.text.values)
            # features = df.p_opinion.values
            vec = TfidfVectorizer(lowercase=False, tokenizer=lambda x: x, preprocessor=lambda x: x, min_df=5,
                                  ngram_range=(1, 1))

            b_vec = CountVectorizer(lowercase=False, tokenizer=lambda x: x, preprocessor=lambda x: x, min_df=5,
                                  ngram_range=(1, 1), binary=True)

            c_vec = CountVectorizer(lowercase=False, tokenizer=lambda x: x, preprocessor=lambda x: x, min_df=5,
                                  ngram_range=(1, 1))

            bdata = b_vec.fit_transform(data)
            # bdata = hstack([bdata, features.reshape(-1, 1)]).tocsr()
            cdata = c_vec.fit_transform(data)
            # cdata = hstack([cdata, features.reshape(-1, 1)]).tocsr()
            data = vec.fit_transform(data)
            # data = hstack([data, features.reshape(-1, 1)]).tocsr()

            print(len(vec.vocabulary_), df.shape[0])
            labels = df.sentiment.values if TASK == 'sen' else df.relevant.values

            svm_param_dists = {
                'C': (1e-6, 2),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2, 3],
                'gamma': ['scale', 'auto'],
            }
            svm_cls = tune_model(SVC(random_state=0, class_weight='balanced'), svm_param_dists, data, labels, TASK,
                                rewrite=False)

            nb_param_dists = {
                'alpha': (0.01, 5),
                'fit_prior': [True, False],
            }
            multiNB_cls = tune_model(MultinomialNB(), nb_param_dists, cdata, labels, TASK)
            bernNB_cls = tune_model(BernoulliNB(binarize=None), nb_param_dists, bdata, labels, TASK)

            rf_param_dists = {
                'n_estimators': (10, 256),
                'criterion': ["gini", "entropy"],
                'min_samples_split': (2, 16),
                'max_features': ["sqrt", "log2", None],
                'class_weight': ["balanced", "balanced_subsample", None],
            }
            rf_cls = tune_model(RandomForestClassifier(random_state=0), rf_param_dists, data, labels, TASK)

            gb_param_dists = {
                'learning_rate': (1e-5, 1e-1),
                'n_estimators': (10, 256),
                'subsample': (0.1, 1),
                'criterion': ("friedman_mse", "mse"),
                'min_samples_split': (2, 16),
                "max_depth": (2, 16),
                'max_features': ["sqrt", "log2", None]
            }
            gb_cls = tune_model(GradientBoostingClassifier(random_state=0), gb_param_dists, data, labels, TASK)

            dummy = DummyClassifier(random_state=0, strategy='most_frequent')

            dummy.fit(data, labels)

            df = pd.read_csv(f'mydata/labelled/{t}/{t}_{TASK}_test.tsv', index_col=['text_id'], quoting=3, sep='\t')
            data = prepare_data(df.text.values)
            bdata = b_vec.transform(data)
            # bdata = hstack([bdata, features.reshape(-1, 1)]).tocsr()
            cdata = c_vec.transform(data)
            # cdata = hstack([cdata, features.reshape(-1, 1)]).tocsr()
            data = vec.transform(data)
            # data = hstack([data, features.reshape(-1, 1)]).tocsr()

            print('loaded test:', df.shape[0])
            labels = df.sentiment.values if TASK == 'sen' else df.relevant.values

            results = {}
            for cls in [svm_cls, multiNB_cls, bernNB_cls, rf_cls, gb_cls, dummy]:
                model_name = type(cls).__name__
                if cls is bernNB_cls:
                    print('bern')
                    scores = test(cls, bdata, labels)
                elif cls is multiNB_cls:
                    print('multi')
                    scores = test(cls, cdata, labels)
                else:
                    scores = test(cls, data, labels)
                results[model_name] = scores

            write_results(results, theme=t, TASK=TASK)
