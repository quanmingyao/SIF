import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import Network_Single, Network_Single_Triple, NAS, NAS_Triple
import logging
from time import time
import math

def train_search(train_queue, valid_queue, model, optimizer, arch_optimizer, args):
	torch.manual_seed(args.seed)
	users_train, items_train, labels_train = train_queue
	users_valid, items_valid, labels_valid = valid_queue
	
	optimizer.zero_grad()
	model.zero_grad()
	inferences, regs = model(users_train, items_train)
	loss = model.compute_loss(inferences, labels_train, regs)
	loss.backward()
	optimizer.step()

	if args.mode == 'traindarts':
		loss_valid = model.step(users_train, items_train, labels_train, users_train, 
			items_train, labels_train, args.lr, arch_optimizer, args.unrolled)
	else:
		loss_valid = model.step(users_train, items_train, labels_train, users_valid, 
			items_valid, labels_valid, args.lr, arch_optimizer, args.unrolled)

	g, gp = model.genotype()
	return g, gp, loss.cpu().detach().numpy().tolist(), loss_valid.cpu().detach().numpy().tolist()

def train_search_triple(train_queue, valid_queue, model, optimizer, arch_optimizer, args):
	torch.manual_seed(args.seed)
	p_train, q_train, r_train, labels_train = train_queue
	p_valid, q_valid, r_valid, labels_valid = valid_queue
	
	optimizer.zero_grad()
	model.zero_grad()
	inferences, regs = model(p_train, q_train, r_train)
	loss = model.compute_loss(inferences, labels_train, regs)
	loss.backward()
	optimizer.step()

	loss_valid = model.step(p_train, q_train, r_train, labels_train, p_valid, 
		q_valid, r_valid, labels_valid, args.lr, arch_optimizer, args.unrolled)

	g, gp = model.genotype()
	return g, gp, loss.cpu().detach().numpy().tolist(), loss_valid.cpu().detach().numpy().tolist()

def get_arch_performance(arch, num_users, num_items, train_queue, test_queue, args):
	torch.manual_seed(args.seed)
	model = Network_Single(num_users, num_items, args.embedding_dim, arch, args.weight_decay).cuda()
	optimizer = torch.optim.Adagrad(model.parameters(), args.lr)

	losses = []
	start = time()
	for train_epoch in range(args.train_epochs):
		loss = train_single(train_queue, model, optimizer, args)
		losses.append(loss)

		if train_epoch > 100:
			if (losses[-2]-losses[-1])/losses[-1] < 1e-4/train_queue[0].shape[0] or np.isnan(losses[-1]):
				break
		if args.mode == 'test':
			logging.info('train_epoch: %d, loss: %.4f, rmse: %.4f[%.4f]' % (
				train_epoch, loss, evaluate(model, test_queue), time()-start))
	return evaluate(model, test_queue)

def get_arch_performance_nas(arch, num_users, num_items, train_queue, test_queue, args):
	torch.manual_seed(args.seed)
	model = NAS(num_users, num_items, args.embedding_dim, arch, args.weight_decay).cuda()
	optimizer = torch.optim.Adagrad(model.parameters(), args.lr)

	losses = []
	start = time()
	for train_epoch in range(args.train_epochs):
		loss = train_single(train_queue, model, optimizer, args)
		losses.append(loss)
		if train_epoch > 100:
			if (losses[-2]-losses[-1])/losses[-1] < 1e-4/train_queue[0].shape[0] or np.isnan(losses[-1]):
				break
	return evaluate(model, test_queue)

def get_arch_performance_triple(arch, num_ps, num_qs, num_rs, train_queue, test_queue, args):
	torch.manual_seed(args.seed)
	model = Network_Single_Triple(num_ps, num_qs, num_rs, args.embedding_dim, arch, args.weight_decay).cuda()
	optimizer = torch.optim.Adagrad(model.parameters(), args.lr)

	losses = []
	start = time()
	for train_epoch in range(args.train_epochs):
		loss = train_single_triple(train_queue, model, optimizer, args)
		losses.append(loss)

		if train_epoch > 100:
			if (losses[-2]-losses[-1])/losses[-1] < 1e-4/train_queue[0].shape[0] or np.isnan(losses[-1]):
				break
		if args.mode == 'test':
			logging.info('train_epoch: %d, loss: %.4f, rmse: %.4f[%.4f]' % (
				train_epoch, loss, evaluate(model, test_queue), time()-start))
	return evaluate_triple(model, test_queue)

def get_arch_performance_nas_triple(arch, num_ps, num_qs, num_rs, train_queue, test_queue, args):
	torch.manual_seed(args.seed)
	model = NAS_Triple(num_ps, num_qs, num_rs, args.embedding_dim, arch, args.weight_decay).cuda()
	optimizer = torch.optim.Adagrad(model.parameters(), args.lr)

	losses = []
	start = time()
	for train_epoch in range(args.train_epochs):
		loss = train_single_triple(train_queue, model, optimizer, args)
		losses.append(loss)
		if train_epoch > 100:
			if (losses[-2]-losses[-1])/losses[-1] < 1e-4/train_queue[0].shape[0] or np.isnan(losses[-1]):
				break
	return evaluate_triple(model, test_queue)

def train_single(train_queue, model, optimizer, args):
	torch.manual_seed(args.seed)
	users_train, items_train, labels_train = train_queue
	model.train()
	optimizer.zero_grad()
	model.zero_grad()
	inferences, regs = model(users_train, items_train)
	loss = model.compute_loss(inferences, labels_train, regs)
	loss.backward()
	optimizer.step()
	return loss.cpu().detach().numpy().tolist()

def train_single_minibatch(train_queue, model, optimizer, args):
	torch.manual_seed(args.seed)	
	losses = []
	for (users_train, items_train, labels_train) in train_queue:
		model.train()
		optimizer.zero_grad()
		model.zero_grad()
		inferences, regs = model(users_train.cuda(), items_train.cuda())
		loss = model.compute_loss(inferences, labels_train.cuda(), regs)
		loss.backward()
		optimizer.step()
		losses.append(loss.cpu().detach().numpy().tolist())
	return np.mean(losses)

def train_single_triple(train_queue, model, optimizer, args):
	torch.manual_seed(args.seed)
	ps_train, qs_train, rs_train, labels_train = train_queue
	model.train()
	optimizer.zero_grad()
	model.zero_grad()
	inferences, regs = model(ps_train, qs_train, rs_train)
	loss = model.compute_loss(inferences, labels_train, regs)
	loss.backward()
	optimizer.step()
	return loss.cpu().detach().numpy().tolist()

def train_single_triple_minibatch(train_queue, model, optimizer, args):
	torch.manual_seed(args.seed)	
	losses = []
	for (ps_train, qs_train, rs_train, labels_train) in train_queue:
		model.train()
		optimizer.zero_grad()
		model.zero_grad()
		inferences, regs = model(ps_train.cuda(), qs_train.cuda(), rs_train.cuda())
		loss = model.compute_loss(inferences, labels_train.cuda(), regs)
		loss.backward()
		optimizer.step()
		losses.append(loss.cpu().detach().numpy().tolist())
	return np.mean(losses)

def evaluate(model, test_queue):
	model.eval()

	with torch.no_grad():
		users, items, labels = test_queue
		inferences, _ = model(users, items)
		mse = F.mse_loss(inferences, torch.reshape(labels,[-1,1]))
		rmse = torch.sqrt(mse)
	return rmse.cpu().detach().numpy().tolist()

def evaluate_triple(model, test_queue):
	model.eval()

	with torch.no_grad():
		ps, qs, rs, labels = test_queue
		inferences, _ = model(ps, qs, rs)
		mse = F.mse_loss(inferences, torch.reshape(labels,[-1,1]))
		rmse = torch.sqrt(mse)
	return rmse.cpu().detach().numpy().tolist()

def evaluate_hr_ndcg(model, test_queue, topk=10):
	model.eval()

	with torch.no_grad():
		users, items, _ = test_queue
		users = users.cpu().tolist()
		hrs, ndcgs = [], []
		
		inferences_dict = {}
		
		users_all, items_all = [], []
		for user in list(set(users)):
			users_all += [user] * model.num_items
			items_all += list(range(model.num_items))
		inferences, _ = model(torch.tensor(users_all).cuda(), torch.tensor(items_all).cuda())
		inferences = inferences.detach().cpu().tolist()
		for i, user in enumerate(list(set(users))):
			inferences_dict[user] = inferences[i*model.num_items:(i+1)*model.num_items]

		for i, user in enumerate(users):
			inferences = inferences_dict[user]
			score = inferences[items[i]]
			rank = 0
			for s in inferences:
				if score < s:
					rank += 1
			if rank < topk:
				hr = 1.0
				ndcg = math.log(2) / math.log(rank+2)
			else:
				hr = 0.0
				ndcg = 0.0
			hrs.append(hr)
			ndcgs.append(ndcg)
	return np.mean(hrs), np.mean(ndcgs)

			
			


























