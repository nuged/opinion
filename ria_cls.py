# TODO: bertcls -> train and save model, load it here; try RuBERT and conversational
from bertcls import Classifier, myDataset, device, apply_model
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


model = Classifier()
model.to(device)
model.load_state_dict(torch.load('models/classifier.pt'))
model.eval()

data = open('mydata/ria_comments.txt').read().split('\n')[:-1]
ds = myDataset(range(len(data)), data, [1]*len(data))
dl = DataLoader(ds, batch_size=64, shuffle=False)

scores = {}
for texts, _ in dl:
    with torch.no_grad():
        output = apply_model(model, texts).detach()
        output = nn.functional.softmax(output, dim=1)[:, 1].cpu().numpy()
        answers = output > 0.5
    for i, t in enumerate(texts):
        scores[t] = (output[i], answers[i])

        # loss = criterion(output, labels.to(device).view(-1))
        # pred = output.argmax(axis=1).detach().tolist()
f = open('mydata/ria_scores.tsv', 'w')
for t, (score, answer) in sorted(scores.items(), key=lambda x: x[1][0], reverse=True):
    print(f'{t}\t{score}\t{answer}', file=f)
f.close()
