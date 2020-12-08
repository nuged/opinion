from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_preparation import read_data, remove_emoji, remove_links, remove_duplicates
from multiprocessing import Pool
import torch
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import torch.nn.functional as F
import pymorphy2
from collections import defaultdict


class myDataset(Dataset):
    def __init__(self, ids, texts, labels):
        super(myDataset, self).__init__()
        self.X = [texts[i] for i in ids]
        self.y = [labels[i] for i in ids]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")
        #self.bert = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
        # for p in self.bert.parameters():
            # p.requires_grad = False

        self.config = self.bert.config
        self.config.num_labels = 2
        self.config.max_position_embeddings = 200
        self.fc = nn.Linear(self.config.hidden_size, 1)

    def forward(self, *args, **kwargs):
        x = self.bert(*args, **kwargs)[0][:, 0, :]
        x = nn.ReLU()(x)
        x = self.fc(x)
        return x

tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", do_lower_case=False)


def tocuda(d):
    for k in d:
        d[k] = d[k].cuda()


def train(nepochs, model, dl, criterion, opt, test_dl=None):
    loss_train = []
    test_results = defaultdict(list)

    for epoch in range(nepochs):
        loss_history = []
        model.train()
        # print(f"\nEpoch {epoch}:")
        running_loss = 0
        for i, (text, labels) in enumerate(dl):
            labels = labels.cuda()
            opt.zero_grad()
            input_data = tokenizer(text, padding=True, return_tensors='pt')
            tocuda(input_data)
            output = model(**input_data)
            loss = criterion(output, labels.view(-1, 1).type_as(output))
            loss.backward()
            opt.step()
            loss_history.append(loss.item())
            running_loss += loss.item()
            if i % 10 == 9:
                # print(f"iter {i}, loss={running_loss / 10:3.2f}")
                running_loss = 0
        loss_train.append(loss_history)

        if test_dl is not None:
            test_loss, y_true, y_pred = eval(model, test_dl, criterion)
            acc = accuracy_score(y_true, y_pred) * 100
            pr = precision_score(y_true, y_pred) * 100
            rec = recall_score(y_true, y_pred) * 100
            f1 = f1_score(y_true, y_pred) * 100
            test_results['loss'].append(test_loss)
            test_results['accuracy'].append(acc)
            test_results['precision'].append(pr)
            test_results['recall'].append(rec)
            test_results['F1'].append(f1)
    if test_dl is None:
        return loss_history
    else:
        return loss_history, test_results


def eval(model, dl, criterion):
    model.eval()
    mean_loss = 0
    count = 0
    y_pred = []
    y_true = []
    for text, labels in dl:
        with torch.no_grad():
            labels = labels.cuda()
            input_data = tokenizer(text, padding=True, return_tensors='pt')
            tocuda(input_data)
            output = model(**input_data)
            loss = criterion(output, labels.view(-1, 1).type_as(output))
            mean_loss += loss.item()
            count += 1
            pred = (output > 0).detach().tolist()
            y_pred.extend(pred)
            y_true.extend((labels.tolist()))
    return mean_loss / count, y_true, y_pred

NFOLDS = 4
def CV(data, labels, n_epochs, lr):
    kf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=7)
    criterion = nn.BCEWithLogitsLoss()

    results = {m : np.zeros(n_epochs) for m in ['loss', 'accuracy', 'precision', 'recall', 'F1']}
    for fold, (train_ids, test_ids) in enumerate(kf.split(data, labels)):
        print(f"Fold {fold}")
        torch.random.manual_seed(7)
        torch.cuda.manual_seed(7)
        train_ds = myDataset(train_ids, data, labels)
        test_ds = myDataset(test_ids, data, labels)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=32)

        cls = Classifier()
        cls.cuda()
        optimizer = optim.Adam(cls.parameters(), lr=lr)
        # optimizer = optim.SGD(cls.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-5)

        loss_history, test_results = train(n_epochs, cls, train_dl, criterion, optimizer, test_dl)

        print(test_results)

        for metric in test_results:
            results[metric] += np.array(test_results[metric])

        del cls
        del optimizer
        torch.cuda.empty_cache()

    for metric in results:
        results[metric] /= NFOLDS

    print(f"\nn_epochs = {n_epochs}, lr = {lr}")
    for i in range(n_epochs):
        print(f"epoch #{i}:")
        for metric in results:
            print(f'\t{metric} = {results[metric][i]:4.2f}')



morph = pymorphy2.MorphAnalyzer()
with open('opinion/alphabet_rus.txt', encoding='cp1251') as f:
    wlist = set(f.read().strip().split('\n'))


def lower(match):
    word = match.group(0)
    if morph.parse(word.lower())[0].normal_form in wlist:
        return word.lower()
    else:
        return word


def remove_caps(text):
    capitalize = lambda m: m.group(0).capitalize()
    text = re.sub(r'\b[А-ЯЁ]+\b', lower, text)
    text = re.sub(r'^[а-яё]+\b', capitalize, text)
    return text


if __name__ == "__main__":
    data = read_data('pos_c.txt')
    labels = [1] * len(data)
    data.extend(read_data('neg_c.txt'))
    labels += [0] * (len(data) - len(labels))

    p = Pool(processes=4)
    data = p.map(remove_links, data)
    data = p.map(remove_emoji, data)
    data = p.map(str.strip, data)
    p.close()

    for lr in [1e-5]:
        CV(data, labels, 3, lr)

    exit(0)

    cls = Classifier()
    cls.cuda()
    optimizer = optim.Adam(cls.parameters(), lr=1e-5)

    train_ds = myDataset(range(len(data)), data, labels)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    torch.random.manual_seed(7)
    train(3, cls, train_dl, nn.BCEWithLogitsLoss(), optimizer)
    print("trained")
    
    cls.eval()
    scores = []
    answers = []
    with torch.no_grad():
        for text in data:
            input_data = tokenizer(text, return_tensors='pt')
            tocuda(input_data)
            output = cls(**input_data)
            answer = (output > 0).item()
            score = F.sigmoid(output)
            scores.append(score.item())
            answers.append(answer)

    with open("classified.txt", "w") as f:
        for i, text in enumerate(data):
            print(f"{text}\t{answers[i]}\t{scores[i]}", file=f)
    
    exit(0)
    print(len(data))

    cls = Classifier()
    cls.cuda()
    optimizer = optim.Adam(cls.parameters(), lr=1e-5)

    train_ds = myDataset(range(len(data)), data, labels)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    train(4, cls.cuda(), train_dl, nn.BCEWithLogitsLoss(), optimizer)
    print("trained")

    data = read_data("found_nonums.txt")
    p = Pool(processes=4)
    data = p.map(remove_emoji, data)
    data = p.map(remove_links, data)
    data = p.map(str.strip, data)
    data = p.map(remove_caps, data)
    p.close()

    print(data[:5])

    cls.eval()
    scores = []
    answers = []
    with torch.no_grad():
        for text in data:
            input_data = tokenizer(text, return_tensors='pt')
            tocuda(input_data)
            output = cls(**input_data)
            answer = (output > 0).item()
            score = F.sigmoid(output)
            scores.append(score.item())
            answers.append(answer)

    with open("classified.txt", "w") as f:
        for i, text in enumerate(data):
            print(f"{text}\t{answers[i]}\t{scores[i]}", file=f)
