from transformers import BertTokenizer, BertModel, AdamW
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_preparation import *
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from IPython import display
import pickle
from math import ceil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(10)


class myDataset(Dataset):  # here
    def __init__(self, ids, texts, labels):
        super(myDataset, self).__init__()
        self.X = [texts[i] for i in ids]
        self.y = [labels[i] for i in ids]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


class Klassifier(nn.Module):
    def __init__(self, n_outputs=2):
        super(Klassifier, self).__init__()
        self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
        self.nout = n_outputs
        self.config = self.bert.config
        self.fc = nn.Linear(self.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, n_outputs)
        self.drop = nn.Dropout(0.5)
        self.tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-conversational')
        self.tokenizer.model_max_length = 128

    def forward(self, *args, **kwargs):
        x = self.bert(**kwargs).pooler_output # last_hidden_state.mean(axis=1)
        x = self.drop(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

def todevice(d):  # here
    for k in d:
        # print(k, d[k].size())
        d[k] = d[k].to(device)


def apply_model(model, texts):
    tokenizer = model.tokenizer
    if type(texts[0]) is str:
        input_data = tokenizer(texts, padding=True, return_tensors='pt', truncation=True)
        other_features = None
    else:
        text_features = [t for t in texts if type(t[0]) is str]
        other_features = [t.to(device) for t in texts if type(t[0]) is not str]
        input_data = tokenizer(*text_features, padding=True, return_tensors='pt', truncation=True)
    todevice(input_data)
    output = model(other_features, **input_data)
    return output


def train_epoch(model, optimizer, scheduler, train_dl):
    criterion = nn.CrossEntropyLoss()
    model.train()
    loss_history = []
    running_loss = 0
    for i, (texts, labels) in enumerate(train_dl):
        optimizer.zero_grad()
        output = apply_model(model, texts)
        loss = criterion(output, labels.to(device).view(-1))
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_history.append(loss.item())
        running_loss += loss.item()
    return loss_history


def train(model, optimizer, scheduler, train_dl, epochs, test_dl=None):
    model.train()
    avg = 'binary' if model.nout <= 2 else 'macro'
    for epoch in range(epochs):
        epoch_history = train_epoch(model, optimizer, scheduler, train_dl)
        if test_dl is not None:
            test_loss, y_true, probs = predict(model, test_dl)
            y_pred = np.argmax(probs, axis=1)
            scores = {'F1': f1_score(y_true, y_pred, average=avg) * 100,
                      'acc': accuracy_score(y_true, y_pred) * 100}
            yield epoch_history, test_loss, scores, y_true, probs
        else:
            yield epoch_history


def predict(model, dl):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    y_true = []
    probs = []
    mean_loss = 0
    count = 0

    with torch.no_grad():
        for texts, labels in dl:
            output = apply_model(model, texts)
            probs.extend(nn.Softmax(dim=1)(output).detach().tolist())

            loss = criterion(output, labels.to(device).view(-1))
            mean_loss += loss.item()
            count += 1

            y_true.extend((labels.tolist()))
    return mean_loss / count, y_true, probs


def plot_history(loss_history, validation, F1, acc, sizes, title="", clear_output=True): # here
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


def cross_validation(model, params, data, labels, n_splits=4, title=''): # look in colab!
    BS = params['BS']
    EPOCHS = params['n_epochs']

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3)
    probabilities = [[] for e in range(EPOCHS)]
    gts = [[] for e in range(EPOCHS)]
    print(len(gts))

    epoch_size = len(data) / n_splits * (n_splits - 1)
    epoch_size = int(epoch_size)
    epoch_size = (epoch_size + BS - 1) // BS

    scores_avg = {m: np.zeros(EPOCHS) for m in ['F1', 'acc']}  # metric -> [m_at_ep_1, m_at_ep_2, ... ]
    loss_history = np.zeros((EPOCHS, epoch_size))
    val_history = np.zeros(EPOCHS)

    torch.manual_seed(7)

    optimizer = AdamW(model.parameters(), lr=params['lr'], weight_decay=params['w_decay'])

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'models/{type(model).__name__}_init.pt')

    for n, (train_idx, test_idx) in enumerate(cv.split(data, labels)):
        print(f'{n}-th fold')

        if device == 'cuda':
            torch.cuda.empty_cache()

        train_ds = myDataset(train_idx, data, labels)
        test_ds = myDataset(test_idx, data, labels)
        train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=BS, shuffle=False)

        checkpoint = torch.load(f'models/{type(model).__name__}_init.pt')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = OneCycleLR(optimizer, max_lr=params['max_lr'], steps_per_epoch=ceil(len(train_ds) / BS),
                               epochs=EPOCHS, anneal_strategy='linear')
        epoch = 0
        for epoch_history, test_loss, test_scores, y_true, probs in \
                train(model, optimizer, scheduler, train_dl, EPOCHS, test_dl):
            loss_history[epoch] += epoch_history
            val_history[epoch] += test_loss
            # for m, val in test_scores.items():
            # scores_avg[m][epoch] += val=

            # loss, y_test, probs = predict(model, test_dl)
            probabilities[epoch].extend(probs)
            gts[epoch].extend(y_true)
            epoch += 1

    with open(f'history/{title}_{type(model).__name__}_probs_CV.pk', 'wb') as f:
        pickle.dump((gts, probabilities), f)

    f1 = []
    acc = []
    for e in range(EPOCHS):
        y_true = gts[e]
        y_pred = np.argmax(probabilities[e], axis=1)
        f1.append(f1_score(y_true, y_pred, average='binary' if model.nout == 2 else 'macro'))
        acc.append(accuracy_score(y_true, y_pred))

    for m in scores_avg:
        scores_avg[m] /= n_splits
    loss_history /= n_splits
    val_history /= n_splits
    sizes = np.cumsum([epoch_size for i in range(EPOCHS)])

    fig = plot_history(loss_history.reshape(-1), val_history, f1, acc,
                       sizes, title=title, clear_output=False)

    fig.savefig(f'plots/{title}_{type(model).__name__}_plot_CV.png')

    return gts, probabilities

def simple_test(model, params, train_ds, test_ds, title=''):
    torch.manual_seed(7)
    name = type(model).__name__
    BS = params['BS']
    EPOCHS = params['n_epochs']
    optimizer = AdamW(model.parameters(), lr=params['lr'], weight_decay=params['w_decay'])
    scheduler = None if not params['shed'] \
        else OneCycleLR(optimizer, max_lr=params['max_lr'], steps_per_epoch=ceil(len(train_ds) / BS),
                        epochs=EPOCHS, anneal_strategy='linear')

    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BS, shuffle=False)

    loss_history = []
    validation = []
    sizes = []
    F1 = []
    acc = []

    y_history = []
    epoch = 0
    best_f1 = 0
    for epoch_history, test_loss, test_scores, y_true, probs in train(model, optimizer, scheduler, train_dl, EPOCHS,
                                                                       test_dl):
        f1 = test_scores['F1']
        if f1 > best_f1:
            print(f'epoch #{epoch}:\treached better score ({f1:4.2f} vs {best_f1:4.2f})')
            best_f1 = f1
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
            }, f'models/{title}_{name}.pt')

        loss_history.extend(epoch_history)
        validation.append(test_loss)
        sizes.append(len(loss_history))
        F1.append(test_scores['F1'])
        acc.append(test_scores['acc'])
        y_history.append((y_true, probs))

        fig = plot_history(loss_history, validation, F1, acc, sizes, title='')

        epoch += 1

    fig.savefig(f'plots/{title}_{name}_plot.png')
    idx = np.argmax(F1)
    with open(f'history/{title}_{name}_probs.pk', 'wb') as f:
        pickle.dump(y_history[idx], f)

    return y_history[idx]


if __name__ == "__main__":
    data = read_data('mydata/opinion mining/pos_final.txt')
    labels = [1] * len(data)
    data.extend(read_data('mydata/opinion mining/neg_final.txt'))
    labels += [0] * (len(data) - len(labels))

    data = data[:100]
    labels = labels[:100]

    ds = myDataset(range(len(data)), data, labels)
    print(f'loaded {len(ds)} examples')
    train_ds, test_ds = train_test_split(ds, test_size=0.3, shuffle=True, random_state=35,
                                         stratify=labels)

    cls = Klassifier().to(device)

    print(cls.tokenizer)

    params = {
        'n_epochs': 3,
        'BS': 10,
        'lr': 1e-6,
        'max_lr': 1e-2,
        'w_decay': 1e-5,
        'shed': False
    }

    simple_test(cls, params, train_ds, test_ds, title='sentiment_qa')
