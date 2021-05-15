import pandas as pd
from classification import write_results, accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, AdamW
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from bertcls import todevice, myDataset, plot_history
from torch.optim import lr_scheduler
from math import ceil
import gc
from sklearn.metrics import confusion_matrix
import pickle

TASK = 'relevance'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_num_threads(10)


class RuBERT_conv(nn.Module):
    def __init__(self):
        super(RuBERT_conv, self).__init__()
        self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")

        self.config = self.bert.config
        self.config.max_position_embeddings = 256
        self.fc = nn.Linear(self.config.hidden_size, 512)
        n_outputs = 2 if TASK == 'relevance' else 3
        self.fc2 = nn.Linear(512, n_outputs)
        self.drop = nn.Dropout(0.5)

    def forward(self, *args, **kwargs):
        x = self.bert(*args, **kwargs).last_hidden_state.mean(axis=1)
        x = nn.ReLU()(x)
        x = self.drop(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x


def apply_model(model, texts):
    tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence", do_lower_case=False)
    if type(texts[0]) is str:
        input_data = tokenizer(texts, padding=True, return_tensors='pt')
    else:
        input_data = tokenizer(*texts, padding=True, return_tensors='pt')
    todevice(input_data)
    output = model(**input_data)
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
        scheduler.step()
        loss_history.append(loss.item())
        running_loss += loss.item()
    return loss_history


def train(model, optimizer, scheduler, train_dl, epochs, test_dl=None):
    model.train()
    avg = 'binary' if TASK == 'relevance' else 'macro'
    for epoch in range(epochs):
        epoch_history = train_epoch(model, optimizer, scheduler, train_dl)
        if test_dl is not None:
            test_loss, y_true, y_pred = predict(model, test_dl)
            scores = {'F1': f1_score(y_true, y_pred, average=avg),
                      'acc': accuracy_score(y_true, y_pred)}
            yield epoch_history, test_loss, scores, confusion_matrix(y_true, y_pred)
        else:
            yield epoch_history


def predict(model, dl):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    y_true = []
    y_pred = []
    mean_loss = 0
    count = 0

    with torch.no_grad():
        for i, (texts, labels) in enumerate(dl):
            output = apply_model(model, texts)

            loss = criterion(output, labels.to(device).view(-1))
            mean_loss += loss.item()
            count += 1

            pred = output.argmax(axis=1).detach().tolist()
            y_pred.extend(pred)
            y_true.extend((labels.tolist()))

    return mean_loss/count, y_true, y_pred


def simple_test(model, train_ds, test_ds, theme=''):
    torch.manual_seed(7)
    BS = 32
    EPOCHS = 1
    optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=1e-6)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=7e-5, steps_per_epoch=ceil(len(train_ds) / BS),
                                        epochs=EPOCHS)
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BS, shuffle=False)


    loss_history = []
    validation = []
    sizes = []
    F1 = []
    acc = []

    for epoch_history, test_loss, test_scores, confmat in train(model, optimizer, scheduler, train_dl, EPOCHS, test_dl):
        loss_history.extend(epoch_history)
        validation.append(test_loss)
        sizes.append(len(loss_history))
        F1.append(test_scores['F1'])
        acc.append(test_scores['acc'])

        fig = plot_history(loss_history, validation, F1, acc, sizes, title=theme)

    with open(f'{TASK}_{theme}_{type(model).__name__}.pk', 'wb') as f:
        pickle.dump(confmat, f)


def cross_validation(model, data, labels, n_splits=4):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3)
    predictions = []
    gts = []

    optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=1e-6)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'models/{type(model).__name__}_cv.pt')

    for n, (train_idx, test_idx) in enumerate(cv.split(data, labels)):
        print(f'{n}-th fold')
        torch.manual_seed(7)

        if device == 'cuda':
            torch.cuda.empty_cache()

        train_ds = myDataset(train_idx, data, labels)
        test_ds = myDataset(test_idx, data, labels)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

        checkpoint = torch.load(f'models/{type(model).__name__}_cv.pt')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=7e-5, steps_per_epoch=ceil(len(train_ds) / 32),
                                            epochs=8)

        train(model, optimizer, scheduler, train_dl, epochs=8)
        y_test, y_pred = predict(model, test_dl)
        predictions.extend(y_pred)
        gts.extend(y_test)

        del train_ds
        del train_dl
        del test_ds
        del test_dl
        gc.collect()

        if device == 'cuda':
            torch.cuda.empty_cache()

    conf = confusion_matrix(gts, predictions)

    return conf


if __name__ == '__main__':
    df = pd.read_csv(f'mydata/labelled/relevance_nli.tsv', quoting=3, sep='\t')
    data = df[['text', 'hypothesis']].values.tolist()
    labels = df.label.tolist()
    ds = myDataset(range(len(data)), data, labels)
    dl = DataLoader(ds, batch_size=3, shuffle=True)
    tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence", do_lower_case=False)

    model = RuBERT_conv()

    for texts, labels in dl:
        k = apply_model(model, texts)
        print(k.size())

    exit(0)
    for t in ['masks', 'quarantine', 'government', 'vaccines']:
        print(f'{t}:')
        df = pd.read_csv(f'mydata/labelled/{t}/{t}_{TASK}.tsv', index_col=['text_id'], quoting=3, sep='\t')
        data = df.text.values
        labels = df.sentiment.values if TASK == 'sentiment' else df.relevant.values

        results = {}

        ruber_conv = RuBERT_conv().to(device)

        ds = myDataset(range(len(data[:100])), data[:100], labels[:100])
        train_ds, test_ds = train_test_split(ds, test_size=0.3, shuffle=True, random_state=777,
                                             stratify=labels[:100])

        simple_test(ruber_conv, train_ds, test_ds, theme=t)
