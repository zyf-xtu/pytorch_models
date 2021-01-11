#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/21 16:47
# @Author  : zyf
# @File    : test.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Rnn(nn.Module):

    def __init__(self):
        super(Rnn, self).__init__()
        self.seq_len = 30
        self.rnn = nn.LSTM(29, 128, batch_first=True,
                          num_layers=1, dropout=0, bidirectional=False)

        self.fc = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.ELU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            nn.Linear(32, 6),
        )

    def forward(self, x):
        x = x.view(x.size(0), 30, 29)
        h0 = torch.zeros(1, x.size(0), 128).to(device)
        c0 = torch.zeros(1, x.size(0), 128).to(device)
        rnn_out, _ = self.rnn(x, (h0, c0))
        rnn_out = rnn_out[:, -1, :]
        out = self.fc(rnn_out)

        return out


class TestDataset(Dataset):
    def __init__(self, sample_x, sample_y):
        self.sample_x = sample_x
        self.sample_y = sample_y

    def __len__(self):
        return len(self.sample_x)

    def __getitem__(self, idx):
        return self.sample_x[idx], self.sample_y[idx]


sample_x = torch.randn(1000, 30, 29).float().to(device)
sample_y = torch.randint(low=0, high=4, size=(1000, 1)).long().to(device)
test_dataset = TestDataset(sample_x, sample_y)
dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

model = Rnn().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(1000):
    for x_batch, y_batch in dataloader:
        # Forward pass
        model = model.train()
        output = model(x_batch)
        loss = criterion(output, y_batch.reshape(-1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch: %d, Loss: %.3f" % (epoch+1, loss.item()))

    with torch.no_grad():
        model = model.eval()
        output = model(sample_x)
        acc = (output.argmax(axis=1) == sample_y.squeeze()).sum().item() / len(test_dataset)

        print("Epoch: %d, Accuracy: %.3f" % (epoch + 1, acc))