from transformers import BertTokenizer, BertForSequenceClassification, BertModel, AdamW
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_preparation import read_data, remove_emoji, remove_links, remove_duplicates
from multiprocessing import Pool
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
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
        self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")
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


tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", do_lower_case=False)


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


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def CV(data, labels, nfolds=4, train_epochs=3, lr=1e-6, bs=32, wd=1e-6):
    # TODO: добавить отрисовку и сохранение графика усредненных по всем фолдам потерь на тесте и обучении
    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=7)
    criterion = nn.CrossEntropyLoss()
    scores_avg = [defaultdict(list) for i in range(train_epochs)]  # epoch -> {metric -> [val_1, ..., val_nfolds]}
    for fold, (train_ids, test_ids) in enumerate(kf.split(data, labels)):
        torch.manual_seed(7)
        cls = Classifier().to(device)
        optimizer = AdamW(cls.parameters(), lr=lr, weight_decay=wd)

        train_ds = myDataset(train_ids, data, labels)
        test_ds = myDataset(test_ids, data, labels)
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=bs)

        epoch = 0
        loss_history = []
        validation = []
        sizes = []
        for epoch_history, test_loss, test_scores in train(train_epochs, cls, train_dl,
                                                           criterion, optimizer, test_dl):
            epoch_history = moving_average(epoch_history, 10).tolist()
            loss_history.extend(epoch_history)
            validation.append(test_loss)
            sizes.append(len(loss_history))

            if fold == 0:
                plt.plot(np.arange(1, sizes[-1] + 1), loss_history)
                plt.scatter(sizes, validation, marker='*', c='red')
                plt.grid()
                plt.xticks(np.arange(1, sizes[-1] + 50, 50))
                plt.savefig(f'plots/plot_{lr}_{bs}_{wd}_{epoch}.png')
                plt.show()

            for m, val in test_scores.items():
                scores_avg[epoch][m].append(val)
            scores_avg[epoch]['val_loss'].append(test_loss)
            epoch += 1
        if device == 'cuda':
            torch.cuda.empty_cache()

    for epoch in range(train_epochs):
        print(f"epoch #{epoch}")
        scores = scores_avg[epoch]
        print(f"\tval_loss={np.mean(scores['val_loss']):4.3f}")
        del scores['val_loss']
        for m, val in scores.items():
            print(f"\t{m}={np.mean(val):4.2f}")


def simple_test(cls, optim_cls, data, labels, train_epochs=3, lr=1e-6, bs=32, wd=1e-6, title=""):
    import pylab as pl
    from IPython import display

    ds = myDataset(range(len(data)), data, labels)
    train_ds, test_ds = train_test_split(ds, test_size=0.2, shuffle=True, random_state=7,
                                                            stratify=labels)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=bs)

    torch.manual_seed(7)
    optimizer = optim_cls(cls.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    epoch = 0
    loss_history = []
    validation = []
    sizes = []

    best_scores = {m: 0 for m in ['accuracy', 'precision', 'recall', 'F1', 'val_loss']}
    best_epoch = best_scores.copy()
    best_scores['val_loss'] = 1000

    log = []

    F1 = []
    acc = []

    for epoch_history, test_loss, test_scores in train(train_epochs, cls, train_dl,
                                                       criterion, optimizer, test_dl):
        epoch_history = moving_average(epoch_history, 10).tolist()
        log.append(test_scores)
        loss_history.extend(epoch_history)
        validation.append(test_loss)
        sizes.append(len(loss_history))
        F1.append(test_scores['F1'])
        acc.append(test_scores['accuracy'])

        display.clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))
        ax1.plot(np.arange(1, sizes[-1] + 1), loss_history)
        ax1.scatter(sizes, validation, marker='*', c='red', zorder=1)
        ax1.grid()
        ax1.set_xticks([0] + sizes)
        ax1.set_title(f"{lr}_{bs}_{wd}_{title}_{epoch}")
        ax2.scatter(np.arange(0, len(F1)), F1)
        ax2.scatter(np.arange(0, len(F1)), acc)
        ax2.plot(F1, label="F1")
        ax2.plot(acc, label="acc")
        ax2.grid()
        ax2.set_yticks(np.arange(min(F1 + acc), 105, 5))
        ax2.legend()
        # fig = plt.gcf()
        plt.show()

        for m, val in test_scores.items():
            if val > best_scores[m]:
                best_scores[m] = val
                best_epoch[m] = epoch
        if test_loss < best_scores['val_loss']:
            best_scores['val_loss'] = test_loss
            best_epoch['val_loss'] = epoch

        epoch += 1

        # print(f"epoch #{epoch}")
        # print(f"\tval_loss={test_loss:4.3f}")
        # for m, val in test_scores.items():
        #     print(f"\t{m}={val:4.2f}")

    fig.savefig(f'plots/plot_{lr}_{bs}_{wd}_{title}.png')
    with open(f"logs/{lr}_{bs}_{wd}_{title}.log", 'w') as f:
        for m, val in best_scores.items():
            print(f"{m} = {val:4.2f}, epoch = {best_epoch[m]}", file=f)
        best = log[best_epoch['val_loss']]
        print(f"scores at epoch={best_epoch['val_loss']}:", file=f)
        for m, val in best.items():
            print(f"{m} = {val:4.2f}", file=f)

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
        for wd in [1e-4]:
            simple_test(Classifier(), AdamW, data[:100], labels[:100], train_epochs=10, lr=lr, bs=16, wd=wd)
