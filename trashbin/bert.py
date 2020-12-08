from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_preparation import read_data, remove_emoji, remove_links
from multiprocessing import Pool
import torch
from sklearn.model_selection import KFold
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import torch.nn.functional as F

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
        self.bert = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
        self.config = self.bert.config
        self.fc = nn.Linear(self.config.hidden_size, 1)

    def forward(self, *args, **kwargs):
        x = self.bert(*args, **kwargs)[1]
        x = self.fc(x)
        return x


tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

def tocuda(d):
    for k in d:
        d[k] = d[k].cuda()


def train(nepochs, model, dl, criterion, opt):
    loss_history = []
    model.train()
    for epoch in range(nepochs):
        for text, labels in dl:
            labels = labels.cuda()
            opt.zero_grad()
            input_data = tokenizer(text, padding=True, return_tensors='pt')
            tocuda(input_data)
            output = model(**input_data)
            loss = criterion(output, labels.view(-1, 1).type_as(output))
            loss.backward()
            opt.step()
            loss_history.append(loss.item())
    return loss_history


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
            pred = (output > 0.5).detach().tolist()
            y_pred.extend(pred)
            y_true.extend((labels.tolist()))
    return mean_loss / count, y_true, y_pred


def CV(data, labels):
    kf = KFold(n_splits=5, shuffle=True, random_state=7)
    criterion = nn.BCEWithLogitsLoss()

    acc = 0
    pr = 0
    rec = 0
    f1 = 0
    loss = 0
    c = 0
    for train_ids, test_ids in kf.split(data, labels):
        c += 1
        torch.random.manual_seed(7)
        train_ds = myDataset(train_ids, data, labels)
        test_ds = myDataset(test_ids, data, labels)
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=16)

        cls = Classifier()
        optimizer = optim.Adam(cls.parameters(), lr=1e-5)

        losses = train(3, cls.cuda(), train_dl, criterion, optimizer)
        plt.plot(np.arange(1, len(losses) * 16 + 1, 16), losses)
        plt.xticks(np.arange(1, len(losses) * 16 + 1, 16))
        plt.savefig(f"train_loss_{c}.png")
        plt.show()

        test_loss, y_true, y_pred = eval(cls, test_dl, criterion)

        pr += precision_score(y_true, y_pred)
        acc += accuracy_score(y_true, y_pred)
        f1 += f1_score(y_true, y_pred)
        rec += recall_score(y_true, y_pred)
        loss += test_loss

    print(f"accuracy = {acc * 100 / 5:4.2f}")
    print(f"precision = {pr * 100 / 5:4.2f}")
    print(f"recall = {rec * 100 / 5:4.2f}")
    print(f"F1 = {f1 * 100 / 5:4.2f}")
    print(f"loss = {loss / 5}")


def remove_caps(text):
    m = re.search(r'^[А-ЯЁ]+\b', text)
    text = re.sub(m.group(), m.group().capitalize(), text)
    while m := re.search(r'\b[А-ЯЁ]+\b', text):
        text = re.sub(m.group(), m.group().lower(), text)
    return text


if __name__ == "__main__":
    data = read_data('pos_chosen.txt')
    labels = [1] * len(data)
    data.extend(read_data('neg_chosen.txt'))
    labels += [0] * (len(data) - len(labels))

    p = Pool(processes=4)
    data = p.map(remove_emoji, data)
    data = p.map(remove_links, data)
    p.close()

    cls = Classifier()
    cls.cuda()
    optimizer = optim.Adam(cls.parameters(), lr=1e-5)

    train_ds = myDataset(range(len(data)), data, labels)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    train(3, cls.cuda(), train_dl, nn.BCEWithLogitsLoss(), optimizer)
    print("trained")

    data = read_data("found_nonums.txt")
    p = Pool(processes=4)
    data = p.map(remove_emoji, data)
    data = p.map(remove_links, data)
    data = p.map(remove_caps, data)
    p.close()

    cls.eval()
    scores = []
    answers = []
    with torch.no_grad():
        for text in data:
            input_data = tokenizer(text)
            tocuda(input_data)
            output = cls(**input_data)
            answer = (output > 0).item()
            score = F.sigmoid(output.item())
            scores.append(score)
            answers.append(answer)

    with open("classified.txt", "w") as f:
        for i, text in data:
            print(f"{text}\t{answers[i]}\t{scores[i]}")
