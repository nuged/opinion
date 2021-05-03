import pandas as pd
from classification import write_results, accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, AdamW
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from bertcls import todevice, myDataset
from torch.optim import lr_scheduler
from math import ceil
import gc
from sklearn.metrics import confusion_matrix

TASK = 'sentiment'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(10)


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
    input_data = tokenizer(texts, padding=True, return_tensors='pt')
    todevice(input_data)
    output = model(**input_data)
    return output


def train(model, optimizer, scheduler, train_dl, epochs):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for i, (texts, labels) in enumerate(train_dl):
            optimizer.zero_grad()
            output = apply_model(model, texts)
            loss = criterion(output, labels.to(device).view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()


def predict(model, dl):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, (texts, labels) in enumerate(dl):
            probs = nn.Softmax(dim=1)(apply_model(model, texts))
            pred = probs.argmax(axis=1).detach().tolist()
            y_pred.extend(pred)
            y_true.extend((labels.tolist()))
    return y_true, y_pred


def cross_validation(ModelCls, data, labels, n_splits=4):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3)
    predictions = []
    gts = []
    for train_idx, test_idx in cv.split(data, labels):
        torch.manual_seed(7)

        if device == 'cuda':
            torch.cuda.empty_cache()

        train_ds = myDataset(train_idx, data, labels)
        test_ds = myDataset(test_idx, data, labels)
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

        model = ModelCls().to(device)
        optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=1e-6)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=7e-5, steps_per_epoch=ceil(len(train_ds) / 16),
                                            epochs=8)

        train(model, optimizer, scheduler, train_dl, epochs=8)
        y_test, y_pred = predict(model, test_dl)
        predictions.extend(y_pred)
        gts.extend(y_test)

        del train_ds
        del train_dl
        del test_ds
        del test_dl
        del model
        del optimizer
        del scheduler
        gc.collect()

        if device == 'cuda':
            torch.cuda.empty_cache()

    conf = confusion_matrix(gts, predictions)

    return conf


if __name__ == '__main__':
    for t in ['masks', 'quarantine', 'government', 'vaccines']:
        df = pd.read_csv(f'mydata/labelled/{t}/{t}_{TASK}.tsv', index_col=['text_id'], quoting=3, sep='\t')
        data = df.text.values
        labels = df.sentiment.values if TASK == 'sentiment' else df.relevant.values

        results = {}

        for Cls in [RuBERT_conv]:
            model_name = Cls.__name__
            conf = cross_validation(Cls, data, labels, n_splits=5)
            results[model_name] = conf

        with open(f'{TASK}_{t}_conf.txt', 'w') as f:
            for name, c in results.items():
                print(f"{name}:\n{c}\n", file=f)
