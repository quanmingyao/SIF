import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.utils import shuffle
import os
from sklearn.preprocessing import StandardScaler

data_path = '/data3/chenxiangning/AutoMC/ml-1m/ratings.dat'
users, items, labels = [], [], []
with open(data_path, 'r') as f:
	for i,line in enumerate(f.readlines()):
		line = line.split('::')
		users.append(int(line[0]) - 1)
		items.append(int(line[1]) - 1)
		labels.append(float(line[2]))
labels = StandardScaler().fit_transform(np.reshape(labels, [-1,1])).flatten().tolist()
users, items, labels = shuffle(users, items, labels)
num_train = int(len(users) * 0.66)
users_train, items_train, labels_train = torch.tensor(users[:num_train]).cuda(), \
	torch.tensor(items[:num_train]).cuda(), torch.tensor(labels[:num_train]).cuda().float()
users_test, items_test, labels_test = torch.tensor(users[num_train:]).cuda(), \
	torch.tensor(items[num_train:]).cuda(), torch.tensor(labels[num_train:]).cuda().float()

class Model(nn.Module):
	def __init__(self, num_users, num_items):
		super(Model, self).__init__()
		self._UE = nn.Embedding(num_users, 2)
		self._IE = nn.Embedding(num_items, 2)
		self._FC = nn.Linear(2, 1)

	def forward(self, users, items):
		ue = self._UE(users)
		ie = self._IE(items)
		return self._FC(ue + ie)

	def compute_loss(self, inferences, labels):
		labels = torch.reshape(labels, [-1,1])
		return F.mse_loss(inferences, labels)

def update_parameters(model, lr):
	for f in m.parameters():
		f.data.sub_(f.grad.data * lr)

def get_loss(model, users, items, labels):
	inferences = m(users, items)
	loss = m.compute_loss(inferences, labels)
	return loss


m = Model(6040, 3952).cuda()
for _ in range(100000):
	lr = 1.0
	m.train()
	m.zero_grad()
	loss = get_loss(m, users_train, items_train, labels_train)
	loss.backward()
	update_parameters(m, lr)

	loss_after = get_loss(m, users_train, items_train, labels_train)
	if loss_after < loss:
		lr *= 1.1
	else:
		for _ in range(10):
			update_parameters(m, lr*(1/1.4 - 1))
			lr /= 1.4
			if get_loss(m, users_train, items_train, labels_train) < loss:
				break

	m.eval()
	mse = get_loss(m, users_test, items_test, labels_test)
	rmse = torch.sqrt(mse)

	print(loss.cpu().detach().numpy(), lr)














