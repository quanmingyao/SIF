import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.utils import shuffle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

class Data(torch.utils.data.Dataset):

	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		return self.data[index]

def get_queue(users, items, labels, batch_size):
	data = list(zip(users, items, labels))
	return torch.utils.data.DataLoader(Data(data), batch_size=batch_size)


def get_data_queue(data_path, args):
	users, items, labels = [], [], []
	if args.dataset == 'ml-100k':
		data_path += 'u.data'
	elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
		data_path += 'ratings.dat'
	elif args.dataset == 'ml-20m':
		data_path += 'ratings.csv'


	if not 'youtube' in args.dataset:

		with open(data_path, 'r') as f:
			for i,line in enumerate(f.readlines()):
				if args.dataset == 'ml-100k':
					line = line.split()
				elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
					line = line.split('::')
				elif args.dataset == 'ml-20m':
					if i == 0: continue
					line = line.split(',')

				users.append(int(line[0]) - 1)
				items.append(int(line[1]) - 1)
				labels.append(float(line[2]))



		# print(np.std(labels))

		labels = StandardScaler().fit_transform(np.reshape(labels, [-1,1])).flatten().tolist()
		
		# print(len(set(users)), max(users), min(users))
		# print(len(set(items)), max(items), min(items))
		# print(len(users))

		users, items, labels = shuffle(users, items, labels)
		num_train = int(len(users) * args.train_portion)
		num_valid = int(len(users) * args.valid_portion)


		if not args.mode == 'libfm':
			if not args.minibatch:
				train_queue = [torch.tensor(users[:num_train]).cuda(), \
							   torch.tensor(items[:num_train]).cuda(), \
							   torch.tensor(labels[:num_train]).cuda().float()]
			else:
				train_queue = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
					torch.tensor(users[:num_train]), torch.tensor(items[:num_train]), 
					torch.tensor(labels[:num_train])), batch_size=8192)

			valid_queue = [torch.tensor(users[num_train:num_train+num_valid]).cuda(), \
						   torch.tensor(items[num_train:num_train+num_valid]).cuda(), \
						   torch.tensor(labels[num_train:num_train+num_valid]).cuda().float()]
			test_queue = [torch.tensor(users[num_train+num_valid:]).cuda(), \
						  torch.tensor(items[num_train+num_valid:]).cuda(), \
						  torch.tensor(labels[num_train+num_valid:]).cuda().float()]

		else:
			train_queue, valid_queue, test_queue = [], [], []
			for i in range(len(users)):
				if i < num_train:
					train_queue.append({'user': str(users[i]), 'item': str(items[i])})
				elif i >= num_train and i < num_train+num_valid:
					valid_queue.append({'user': str(users[i]), 'item': str(items[i])})
				else:
					test_queue.append({'user': str(users[i]), 'item': str(items[i])})
			
			v = DictVectorizer()
			train_queue = [v.fit_transform(train_queue), np.array(labels[:num_train])]
			# valid_queue = [v.transform(train_queue), np.array(labels[num_train:num_train+num_valid])]
			test_queue = [v.transform(test_queue), np.array(labels[num_train+num_valid:])]

	else:
		[ps, qs, rs, labels] = np.load(data_path).tolist()
		labels = StandardScaler().fit_transform(np.reshape(labels, [-1,1])).flatten().tolist()
		ps, qs, rs, labels = shuffle(ps, qs, rs, labels)
		num_train = int(len(ps) * args.train_portion)
		num_valid = int(len(ps) * args.valid_portion)

		# print(len(set(ps)), max(ps), min(ps))
		# print(len(set(qs)), max(qs), min(qs))
		# print(len(set(rs)), max(rs), min(rs))
		# print(len(ps))

		if not args.mode == 'libfm':
			if not args.minibatch:
				train_queue = [torch.tensor(ps[:num_train]).cuda().long(), \
							   torch.tensor(qs[:num_train]).cuda().long(), \
							   torch.tensor(rs[:num_train]).cuda().long(), \
							   torch.tensor(labels[:num_train]).cuda().float()]
			else:
				train_queue = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
					torch.tensor(ps[:num_train]).long(), torch.tensor(qs[:num_train]).long(), 
					torch.tensor(rs[:num_train]).long(), torch.tensor(labels[:num_train])), batch_size=8192)

			valid_queue = [torch.tensor(ps[num_train:num_train+num_valid]).cuda().long(), \
						   torch.tensor(qs[num_train:num_train+num_valid]).cuda().long(), \
						   torch.tensor(rs[num_train:num_train+num_valid]).cuda().long(), \
						   torch.tensor(labels[num_train:num_train+num_valid]).cuda().float()]
			test_queue = [torch.tensor(ps[num_train+num_valid:]).cuda().long(), \
						  torch.tensor(qs[num_train+num_valid:]).cuda().long(), \
						  torch.tensor(rs[num_train+num_valid:]).cuda().long(), \
						  torch.tensor(labels[num_train+num_valid:]).cuda().float()]


		else:
			train_queue, valid_queue, test_queue = [], [], []
			for i in range(len(ps)):
				if i < num_train:
					train_queue.append({'p': str(ps[i]), 'q': str(qs[i]), 'r': str(rs[i])})
				elif i >= num_train and i < num_train+num_valid:
					valid_queue.append({'p': str(ps[i]), 'q': str(qs[i]), 'r': str(rs[i])})
				else:
					test_queue.append({'p': str(ps[i]), 'q': str(qs[i]), 'r': str(rs[i])})

			v = DictVectorizer()
			train_queue = [v.fit_transform(train_queue), np.array(labels[:num_train])]
			# valid_queue = [v.transform(train_queue), np.array(labels[num_train:num_train+num_valid])]
			test_queue = [v.transform(test_queue), np.array(labels[num_train+num_valid:])]

	return train_queue, valid_queue, test_queue






