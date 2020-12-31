# TODO: bertcls -> train and save model, load it here; try RuBERT and conversational
from bertcls import Classifier, myDataset
from torch.utils.data import DataLoader
import torch


model = Classifier()
model.load_state_dict(torch.load('models/classifier.pt'))
model.eval()

data = open('mydata/ria_comments.txt').read().split('\n')[:-1]
ds =