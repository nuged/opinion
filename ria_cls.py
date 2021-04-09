"""
Opinion classification. Creates a new file with opinion scores for each sentence
"""

from bertcls import Classifier, myDataset, device, apply_model
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import BertTokenizer
from collections import defaultdict
import numpy as np

tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence", do_lower_case=False)
tokenizer.add_special_tokens({'additional_special_tokens': ['[USER]']})
model = Classifier()
model.to(device)
model.load_state_dict(torch.load('models/classifier.pt'))
model.eval()

# user_agg = defaultdict(dict)

for week in [1]:
    print(f'week={week}')
    data = open(f'mydata/citations/citations.tsv').read().split('\n')[:-1]
    # data2users = dict([tuple(d.split('\t'))[::-1] for d in data])

    # data = [d.split('\t')[1] for d in data]

    ds = myDataset(range(len(data)), data, [1]*len(data))
    dl = DataLoader(ds, batch_size=64, shuffle=False)

    scores = {}
    for texts, _ in dl:
        with torch.no_grad():
            output = apply_model(model, texts).detach()
            output = nn.functional.softmax(output, dim=1)[:, 1].cpu().numpy()
            # output = np.random.uniform(0, 1, len(texts))
            # answers = output > 0.5
        for i, t in enumerate(texts):
            scores[t] = output[i]
            # user_agg[data2users[t]][t] = (output[i], answers[i])

            # loss = criterion(output, labels.to(device).view(-1))
            # pred = output.argmax(axis=1).detach().tolist()

f = open(f'mydata/citations/opinion_scores.tsv', 'w')
for text, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f'{text}\t{score}', file=f)
f.close()
