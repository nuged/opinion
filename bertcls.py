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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(10)


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
        self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
        # for p in self.bert.parameters():
            # p.requires_grad = False

        self.config = self.bert.config
        self.config.num_labels = 2
        self.config.max_position_embeddings = 256
        self.fc = nn.Linear(self.config.hidden_size, 1)

    def forward(self, *args, **kwargs):
        x = self.bert(*args, **kwargs).pooler_output
        x = self.fc(x)
        return x


tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence", do_lower_case=False)


def todevice(d):
    for k in d:
        d[k] = d[k].to(device)


def train(nepochs, model, dl, criterion, opt, test_dl=None, report_every=50):
    loss_train = []
    test_results = defaultdict(list)

    for epoch in range(nepochs):
        loss_history = []
        model.train()
        # print(f"\nEpoch {epoch}:")
        running_loss = 0
        for i, (text, labels) in enumerate(dl):
            labels = labels.to(device)
            opt.zero_grad()
            input_data = tokenizer(text, padding=True, return_tensors='pt')
            todevice(input_data)
            output = model(**input_data)
            loss = criterion(output, labels.view(-1, 1).type_as(output))
            loss.backward()
            opt.step()
            loss_history.append(loss.item())
            running_loss += loss.item()
            if i % report_every == report_every - 1:
                # print(f"iter {i}, loss={running_loss / report_every:3.2f}")
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
            labels = labels.to(device)
            input_data = tokenizer(text, padding=True, return_tensors='pt')
            todevice(input_data)
            output = model(**input_data)
            loss = criterion(output, labels.view(-1, 1).type_as(output))
            mean_loss += loss.item()
            count += 1
            pred = (output > 0).detach().tolist()
            y_pred.extend(pred)
            y_true.extend((labels.tolist()))
    return mean_loss / count, y_true, y_pred


def CV(data, labels, n_epochs, lr, bs=32, nfolds=4):
    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=7)
    criterion = nn.BCEWithLogitsLoss()

    results = {m : np.zeros(n_epochs) for m in ['loss', 'accuracy', 'precision', 'recall', 'F1']}
    for fold, (train_ids, test_ids) in enumerate(kf.split(data, labels)):
        print(f"Fold {fold}")
        torch.random.manual_seed(7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(7)
        train_ds = myDataset(train_ids, data, labels)
        test_ds = myDataset(test_ids, data, labels)
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=bs)

        cls = Classifier()
        cls.to(device)
        optimizer = optim.Adam(cls.parameters(), lr=lr)

        loss_history, test_results = train(n_epochs, cls, train_dl, criterion, optimizer, test_dl)

        print(test_results)

        for metric in test_results:
            results[metric] += np.array(test_results[metric])

        del cls
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for metric in results:
        results[metric] /= nfolds

    print(f"\nn_epochs = {n_epochs}, lr = {lr}, bs = {bs}")
    for i in range(n_epochs):
        print(f"epoch #{i}:")
        for metric in results:
            print(f'\t{metric} = {results[metric][i]:4.2f}')


morph = pymorphy2.MorphAnalyzer()

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

    for bs in [32]:
        for lr in [5e-6]:
            CV(data, labels, 7, lr, bs=bs, nfolds=5)

    exit(0)

    cls = Classifier()
    cls.to(device)
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
            todevice(input_data)
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
    cls.to(device)
    optimizer = optim.Adam(cls.parameters(), lr=1e-5)

    train_ds = myDataset(range(len(data)), data, labels)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    train(4, cls.to(device), train_dl, nn.BCEWithLogitsLoss(), optimizer)
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
            todevice(input_data)
            output = cls(**input_data)
            answer = (output > 0).item()
            score = F.sigmoid(output)
            scores.append(score.item())
            answers.append(answer)

    with open("classified.txt", "w") as f:
        for i, text in enumerate(data):
            print(f"{text}\t{answers[i]}\t{scores[i]}", file=f)
