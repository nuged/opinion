from data_preparation import make_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from gensim.models import Word2Vec

texts, labels = make_dataset('data/pos.txt', 'data/neg.txt')

vec = TfidfVectorizer(lowercase=False, min_df=5, tokenizer=lambda x: x, preprocessor=lambda x: x)
data = vec.fit_transform(texts)

model = Word2Vec(sentences=texts, size=100, workers=10, sg=1, seed=7, iter=10, min_count=5)
data = np.array([np.sum([model.wv[word] * data[i, vec.vocabulary_[word]]
                         for word in text if word in model.wv and word in vec.vocabulary_])
                 for i, text in enumerate(texts)])
print(data.shape)
print(data[:2])
exit(0)
cls = SVC(C=1, tol=1e-7, kernel='linear', random_state=0, class_weight='balanced')

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
acc = 0
pr = 0
rec = 0
f1 = 0
for train_idx, test_idx in kf.split(data, labels):
    X_train, X_test = data[train_idx], data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    pr += precision_score(y_test, y_pred)
    acc += accuracy_score(y_test, y_pred)
    f1 += f1_score(y_test, y_pred)
    rec += recall_score(y_test, y_pred)

print(f"accuracy = {acc * 100 / 5:4.2f}")
print(f"precision = {pr * 100 / 5:4.2f}")
print(f"recall = {rec * 100 / 5:4.2f}")
print(f"F1 = {f1 * 100 / 5:4.2f}")

exit(0)

cls.fit(data, labels)
print(data.shape)
features = vec.get_feature_names()
coef = cls.coef_.toarray()[0]
with open("topwords.txt", "w") as f:
    for idx in np.abs(coef).argsort()[-1::-1]:
        print(f"{features[idx]}\t{coef[idx]:3.2f}", file=f)
