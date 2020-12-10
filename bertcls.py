from transformers import BertTokenizer, BertForSequenceClassification, BertModel, AdamW
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
        #     p.requires_grad = False

        self.config = self.bert.config
        self.config.max_position_embeddings = 256
        self.fc = nn.Linear(self.config.hidden_size, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, *args, **kwargs):
        x = self.bert(*args, **kwargs).last_hidden_state[:, 0, :]
        x = nn.ReLU()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc2(x)
        return x


tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence", do_lower_case=False)


def todevice(d):
    for k in d:
        d[k] = d[k].to(device)


def apply_model(model, texts):
    input_data = tokenizer(texts, padding=True, return_tensors='pt')
    todevice(input_data)
    output = model(**input_data)
    return output


def get_scores(y_true, y_pred):
    result = {'accuracy': accuracy_score(y_true, y_pred) * 100, 'precision': precision_score(y_true, y_pred) * 100,
              'recall': recall_score(y_true, y_pred) * 100, 'F1': f1_score(y_true, y_pred) * 100}
    return result


def train_epoch(model, train_loader, criterion, optimizer, report_every=0):
    model.train()
    loss_history = []
    running_loss = 0
    for i, (texts, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = apply_model(model, texts)
        loss = criterion(output, labels.to(device).view(-1))
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        running_loss += loss.item()
        if report_every and i % report_every == report_every - 1:
            print(f"iter {i}, loss={running_loss / report_every:3.2f}")
            running_loss = 0
    return loss_history


def train(nepochs, model, train_loader, criterion, optimizer, test_loader=None, report_every=0):
    for epoch in range(nepochs):
        epoch_history = train_epoch(model, train_loader, criterion, optimizer, report_every)
        if test_loader is not None:
            test_loss, y_true, y_pred = eval(model, test_loader, criterion)
            scores = get_scores(y_true, y_pred)
            yield epoch_history, test_loss, scores
        else:
            yield epoch_history


def eval(model, dl, criterion):
    model.eval()
    mean_loss = 0
    count = 0
    y_pred = []
    y_true = []
    for texts, labels in dl:
        with torch.no_grad():
            output = apply_model(model, texts)
            loss = criterion(output, labels.to(device).view(-1))
            mean_loss += loss.item()
            count += 1
            pred = output.argmax(axis=1).detach().tolist()
            y_pred.extend(pred)
            y_true.extend((labels.tolist()))
    return mean_loss / count, y_true, y_pred


def CV(data, labels, nfolds=4, train_epochs=3, lr=1e-6, bs=32, wd=1e-6):
    # TODO: добавить отрисовку и сохранение графика усредненных по всем фолдам потерь на тесте и обучении
    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=7)
    criterion = nn.CrossEntropyLoss()

    fold_dls = {}
    for fold, (train_ids, test_ids) in enumerate(kf.split(data, labels)):
        train_ds = myDataset(train_ids, data, labels)
        test_ds = myDataset(test_ids, data, labels)
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=bs)
        fold_dls[fold] = (train_dl, test_dl)

    # saved_models = {i: f'models/CV_{i}.pt' for i in range(nfolds)}
    saved_models = {i: Classifier() for i in range(nfolds)}
    saved_opts = {i: AdamW(saved_models[i].parameters(), lr=lr, weight_decay=wd) for i in range(nfolds)}
    for epoch in range(train_epochs):
        # history_avg = []
        loss_avg = []
        scores_avg = defaultdict(list)
        for fold in range(nfolds):
            cls = saved_models[fold]
            optimizer = saved_opts[fold]
            train_dl, test_dl = fold_dls[fold]
            epoch_history, test_loss, test_scores = next(train(1, cls, train_dl, criterion, optimizer, test_dl))

            # history_avg.append(epoch_history)
            loss_avg.append(test_loss)
            [scores_avg[m].append(val) for m, val in test_scores.items()]
        # history_avg = np.mean(history_avg, axis=0)
        loss_avg = np.mean(loss_avg)
        scores_avg = {m: np.mean(vals) for m, vals in scores_avg.items()}
        print(f"epoch #{epoch}")
        print(f"\tval_loss={loss_avg:4.3f}")
        for m, val in scores_avg.items():
            print(f"\t{m}={val:4.2f}")
        if device == 'cuda':
            torch.cuda.empty_cache()


# TODO:
# попробовать:
# увеличить дропаут
# вовзращение метрик после каждой эпохи обучения и их усреднение по фолдам, возможно,
# стоит возвращать список с метриками после каждой эпохи
# выключить слои?
# регуляризация

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

    for lr in [2e-6]:
        CV(data, labels, nfolds=4, train_epochs=3, lr=lr, bs=64, wd=0)
