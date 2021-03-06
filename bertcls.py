import random

from transformers import BertTokenizer, BertForSequenceClassification, BertModel, AdamW
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_preparation import *
from multiprocessing import Pool
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
from collections import defaultdict
from IPython import display

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

        # for p in self.bert.named_parameters():
        #     if "layer" in p[0] and int(p[0].split(".")[2]) > 5:
        #         p[1].requires_grad = True

        self.config = self.bert.config
        self.config.max_position_embeddings = 256
        # self.config.hidden_dropout_prob = 0.4
        # self.config.attention_probs_dropout_prob = 0.4
        self.fc = nn.Linear(self.config.hidden_size, 1024)
        self.fc2 = nn.Linear(1024, 2)
        # self.drop = nn.Dropout(0.1)

    def forward(self, *args, **kwargs):
        x = self.bert(*args, **kwargs).last_hidden_state.mean(axis=1)
        x = nn.LeakyReLU(0.1)(x)
        # x = self.drop(x)
        x = self.fc(x)
        x = nn.LeakyReLU(0.1)(x)
        # x = self.drop(x)
        x = self.fc2(x)
        return x


tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence", do_lower_case=False)
tokenizer.model_max_length = 128

def todevice(d):
    for k in d:
        # print(k, d[k].size())
        d[k] = d[k][:, :512].to(device)


def apply_model(model, texts):
    input_data = tokenizer(texts, padding=True, return_tensors='pt', max_length=128, truncation=True)
    todevice(input_data)
    output = model(**input_data)
    return output

def get_scores(y_true, y_pred):
    result = {'accuracy': accuracy_score(y_true, y_pred) * 100, 'precision': precision_score(y_true, y_pred) * 100,
              'recall': recall_score(y_true, y_pred) * 100, 'F1': f1_score(y_true, y_pred) * 100}
    return result


def train_epoch(model, train_loader, criterion, optimizer, report_every=0, scheduler=None):
    model.train()
    loss_history = []
    running_loss = 0
    for i, (texts, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = apply_model(model, texts)
        loss = criterion(output, labels.to(device).view(-1))
        loss.backward()
        optimizer.step()
        if (scheduler):
            scheduler.step()
        loss_history.append(loss.item())
        running_loss += loss.item()
        if report_every and i % report_every == report_every - 1:
            print(f"iter {i}, loss={running_loss / report_every:3.2f}")
            running_loss = 0

    return loss_history


def train(nepochs, model, train_loader, criterion, optimizer, test_loader=None, report_every=0, scheduler=None):
    print('started training')
    for epoch in range(nepochs):
        print(f'epoch #{epoch}')
        epoch_history = train_epoch(model, train_loader, criterion, optimizer, report_every, scheduler)
        if test_loader is not None:
            test_loss, y_true, y_pred, probs = eval(model, test_loader, criterion)
            scores = get_scores(y_true, y_pred)
            yield epoch_history, test_loss, scores, probs
        else:
            yield epoch_history


def eval(model, dl, criterion):
    model.eval()
    mean_loss = 0
    count = 0
    y_pred = []
    y_true = []
    probs = []
    for texts, labels in dl:
        with torch.no_grad():
            output = apply_model(model, texts)
            probs.extend(nn.Softmax(dim=1)(output)[:, 1].detach().tolist())
            loss = criterion(output, labels.to(device).view(-1))
            mean_loss += loss.item()
            count += 1
            pred = output.argmax(axis=1).detach().tolist()
            y_pred.extend(pred)
            y_true.extend((labels.tolist()))
    return mean_loss / count, y_true, y_pred, probs


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_history(loss_history, validation, F1, acc, sizes, title="", clear_output=True):
    if clear_output:
        display.clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))
    ax1.plot(np.arange(1, sizes[-1] + 1), loss_history, zorder=-1, label="Обучение")
    ax1.scatter(sizes, validation, marker='*', c='red', zorder=1, s=50, label='Валидация')
    ax1.grid()
    ax1.set_xticks([0] + sizes)
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('Кросс-энтропия')
    ax1.legend()
    ax2.scatter(np.arange(0, len(F1)), F1)
    ax2.scatter(np.arange(0, len(F1)), acc)
    ax2.plot(F1, label="F1")
    ax2.plot(acc, label="Accuracy")
    ax2.grid()
    ax2.set_yticks(np.linspace(min(F1), max(acc), 20))
    ax2.set_xticks(np.arange(0, len(sizes), 1))
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Метрика (%)')
    ax2.legend()
    fig.suptitle(title)
    plt.show()
    return fig


metrics = ['accuracy', 'precision', 'recall', 'F1']


def CV(data, labels, nfolds=4, train_epochs=3, lr=1e-6, bs=32, wd=1e-6):
    # TODO: добавить отрисовку и сохранение графика усредненных по всем фолдам потерь на тесте и обучении
    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=7)
    criterion = nn.CrossEntropyLoss()

    epoch_size = len(data) / nfolds * (nfolds - 1)
    epoch_size = int(epoch_size)
    epoch_size = (epoch_size + bs - 1) // bs - 9

    scores_avg = {m: np.zeros(train_epochs) for m in metrics}  # metric -> [m_at_ep_1, m_at_ep_2, ... ]
    loss_history = np.zeros((train_epochs, epoch_size))
    val_history = np.zeros(train_epochs)

    for fold, (train_ids, test_ids) in enumerate(kf.split(data, labels)):
        print(f"Fold #{fold}")
        torch.manual_seed(7)
        cls = Classifier().to(device)
        optimizer = AdamW(cls.parameters(), lr=lr, weight_decay=wd)

        train_ds = myDataset(train_ids, data, labels)
        test_ds = myDataset(test_ids, data, labels)
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=bs)

        epoch = 0

        for epoch_history, test_loss, test_scores in train(train_epochs, cls, train_dl,
                                                           criterion, optimizer, test_dl):

            epoch_history = np.array(moving_average(epoch_history, 10))[:epoch_size]
            loss_history[epoch] += epoch_history
            val_history[epoch] += test_loss

            for m, val in test_scores.items():
                scores_avg[m][epoch] += val
            epoch += 1

        if device == 'cuda':
            torch.cuda.empty_cache()

    for m in scores_avg:
        scores_avg[m] /= nfolds
    loss_history /= nfolds
    val_history /= nfolds
    sizes = np.cumsum([epoch_size for i in range(train_epochs)])

    fig = plot_history(loss_history.reshape(-1), val_history, scores_avg['F1'], scores_avg['accuracy'],
                       sizes, title=f"CV: lr={lr}, bs={bs}, wd={wd}", clear_output=False)

    for epoch in range(train_epochs):
        print(f"epoch #{epoch}")
        print(f"\tval_loss={val_history[epoch]:4.3f}")
        for m, val in scores_avg.items():
            print(f"\t{m}={val[epoch]:4.2f}")


def simple_test(cls, optim_cls, data, labels, train_epochs=3, lr=1e-6, bs=32, wd=1e-6, title=""):
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

        fig = plot_history(loss_history, validation, F1, acc, sizes, title)

        for m, val in test_scores.items():
            if val > best_scores[m]:
                best_scores[m] = val
                best_epoch[m] = epoch
        if test_loss < best_scores['val_loss']:
            best_scores['val_loss'] = test_loss
            best_epoch['val_loss'] = epoch

        epoch += 1

    save_results(fig, log, best_scores, best_epoch, title)


def save_results(fig, log, best_scores, best_epoch, title=""):
    fig.savefig(f'plots/{title}.png')
    with open(f"logs/{title}.log", 'w') as f:
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
    data = read_data('mydata/opinion mining/pos_final.txt')
    labels = [1] * len(data)
    data.extend(read_data('mydata/opinion mining/neg_final.txt'))
    labels += [0] * (len(data) - len(labels))
    ds = myDataset(random.sample(range(len(data)), 1000), data, labels)
    exit(0)
    print('loaded', len(data))
    # p = Pool(processes=4)
    # data = p.map(remove_links, data)
    # data = p.map(remove_emoji, data)
    # data = p.map(fix_start, data)
    # data = p.map(fix_quotes, data)
    # data = p.map(remove_sources, data)
    # data = p.map(str.strip, data)
    # p.close()

    cls = Classifier().to(device)

    # simple_test(cls, AdamW, data, labels, 7, lr=1e-5, bs=64, wd=0, title='yeboii')
    torch.manual_seed(5)
    ds = myDataset(range(len(data)), data, labels)
    train_dl = DataLoader(ds, batch_size=64, shuffle=True)
    loss = nn.CrossEntropyLoss()
    opt = AdamW(cls.parameters(), lr=1e-5, weight_decay=0)
    for _ in train(7, cls, train_dl, loss, opt, report_every=10):
        pass
    torch.save(cls.state_dict(), 'models/classifier.pt')
