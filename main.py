import os
import sys
import glob
import numpy as np
import torch
import logging
import argparse
import setproctitle
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch.autograd import Variable
from models import Network, Network_Triple, Network_MLP, Network_MLP_Triple, SPACE, SPACE_NAS
from models import Network_Single, Network_Single_Triple, Network_MLP_Single, Network_MLP_Single_Triple
from models import NCF, DeepWide, AltGrad, Outer, Conv, Plus, Max, Min, ConvNCF
from models import NCF_Triple, DeepWide_Triple, CP, TuckER
from models import AutoNeural, AutoNeural_Triple
from models import PRIMITIVES_UNARY, PRIMITIVES_ASSIST, PRIMITIVES_BINARY, PRIMITIVES_TRIPLE, PRIMITIVES_NAS
from train_eval import train_search, train_single, train_search_triple, train_single_triple
from train_eval import train_single_minibatch, train_single_triple_minibatch
from train_eval import get_arch_performance, get_arch_performance_triple, get_arch_performance_nas, get_arch_performance_nas_triple
from train_eval import evaluate, evaluate_triple, evaluate_hr_ndcg
from dataset import get_data_queue
from time import time
from itertools import chain
from controller import sample_arch, sample_arch_triple, sample_arch_nas, update_arch, Controller, Controller_NAS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


parser = argparse.ArgumentParser(description="Run.")
parser.add_argument('--lr', type=float, default=0.05, help='init learning rate')
parser.add_argument('--arch_lr', type=float, default=0.05, help='learning rate for arch encoding')
parser.add_argument('--controller_lr', type=float, default=1e-1, help='learning rate for controller')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--update_freq', type=int, default=1, help='frequency of updating architeture')
parser.add_argument('--opt', type=str, default='Adagrad', help='choice of opt')
parser.add_argument('--use_gpu', type=int, default=1, help='whether use gpu')
parser.add_argument('--minibatch', type=int, default=0, help='whether use minibatch')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--train_epochs', type=int, default=10000, help='num of training epochs')
parser.add_argument('--search_epochs', type=int, default=1000, help='num of searching epochs')
parser.add_argument('--save', type=str, default='/data3/chenxiangning/logs', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--valid_portion', type=float, default=0.25, help='portion of validation data')
parser.add_argument('--dataset', type=str, default='ml-100k', help='dataset')
parser.add_argument('--mode', type=str, default='hh', help='search or single mode')
parser.add_argument('--process_name', type=str, default='test@xiangning', help='process name')
parser.add_argument('--embedding_dim', type=int, default=2, help='dimension of embedding')
parser.add_argument('--controller', type=str, default='PURE', help='structure of controller')
parser.add_argument('--controller_batch_size', type=int, default=4, help='batch size for updating controller')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--max_batch', type=int, default=65536, help='max batch during training')
args = parser.parse_args()


if __name__ == '__main__':

	torch.set_default_tensor_type(torch.FloatTensor)
	setproctitle.setproctitle(args.process_name)

	log_format = '%(asctime)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, filemode='w',
		format=log_format, datefmt='%m/%d %I:%M:%S %p')

	save_name = 'zzz' + '_' + args.mode + '_' + args.dataset + '_' + str(args.embedding_dim) + \
		'_' + args.opt + str(args.lr)

	if args.mode == 'reinforce':
		save_name += '_' + str(args.controller_lr) + '_' + args.controller + '_' + \
		str(args.controller_batch_size)
	elif args.mode == 'binarydarts':
		save_name += '_' + str(args.arch_lr)
	elif args.mode == 'test' or args.mode == 'autoneural':
		save_name += '_' + str(args.weight_decay)
	else:
		save_name += '_' + str(args.weight_decay)
	save_name += '_' + str(args.seed)

	if not os.path.exists(args.save): os.mkdir(args.save)
	
	if os.path.exists(os.path.join(args.save, save_name + '.txt')):
		os.remove(os.path.join(args.save, save_name + '.txt'))

	fh = logging.FileHandler(os.path.join(args.save, save_name + '.txt'))
	fh.setFormatter(logging.Formatter(log_format))
	logging.getLogger().addHandler(fh)
	
	if args.use_gpu:
		torch.cuda.set_device(args.gpu)
		logging.info('gpu device = %d' % args.gpu)
	else:
		logging.info('no gpu')
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	data_start = time()
	dim = 2
	if not 'youtube' in args.dataset:
		data_path = '/data3/chenxiangning/AutoMC/' + args.dataset + '/'
	elif args.dataset == 'youtube_small':
		data_path = '/data3/chenxiangning/AutoMC/' + 'youtube-weighted_small.npy'
	elif args.dataset == 'youtube':
		data_path = '/data3/chenxiangning/AutoMC/' + 'youtube-weighted.npy'

	if args.dataset == 'ml-100k':
		num_users = 9433
		num_items = 1682
	elif args.dataset == 'ml-1m':
		num_users = 6040
		num_items = 3952
	elif args.dataset == 'ml-10m':
		num_users = 71567
		num_items = 65133
	elif args.dataset == 'ml-20m':
		num_users = 138493
		num_items = 131262
	elif args.dataset == 'youtube_small':
		num_ps = 600
		num_qs = 14340
		num_rs = 5
		dim = 3
	elif args.dataset == 'youtube':
		num_ps = 15088
		num_qs = 15088
		num_rs = 5
		dim = 3
	train_queue, valid_queue, test_queue = get_data_queue(data_path, args)
	logging.info('prepare data finish! [%f]' % (time()-data_start))


	if args.mode == 'reinforce':
		search_start = time()
		performance = {}
		best_arch, best_rmse = None, 100000
		if dim == 2:
			controller = Controller(args.controller).cuda()
		elif dim == 3:
			controller = Controller(args.controller, triple=True).cuda()
		if args.controller == 'RNN':
			optimizer = torch.optim.Adam(controller.parameters(), args.controller_lr)
		elif args.controller == 'PURE':
			optimizer = torch.optim.Adam(controller.arch_parameters(), args.controller_lr)
		for search_epoch in range(args.search_epochs):
			if len(performance.keys()) == 200: break

			archs = controller.sample_arch(args.controller_batch_size)
			rewards = []
			for arch in archs:
				if str(arch) in performance.keys():
					rmse = performance[str(arch)]
				else:
					arch_start = time()
					if dim == 2:
						rmse = get_arch_performance(arch, num_users, num_items, train_queue, test_queue, args)
					elif dim == 3:
						rmse = get_arch_performance_triple(arch, num_ps, num_qs, num_rs, train_queue, test_queue, args)
					performance[str(arch)] = rmse
					logging.info('search_epoch: %d, arch: %s, rmse: %.4f, arch spent: %.4f' % (
						search_epoch, arch, rmse, time()-arch_start))

				if rmse < best_rmse: best_arch, best_rmse = arch, rmse
				if np.isnan(rmse): 
					rewards.append(0.0)
				else:
					rewards.append(1.0 / (rmse + 1e-5))

			optimizer.zero_grad()
			controller.zero_grad()
			loss = controller.compute_loss(rewards)
			loss.backward()
			optimizer.step()
			logging.info('search_epoch: %d, best_arch: %s, best_rmse: %.4f, loss: %.4f, time_spent: %.4f' % (
				search_epoch, best_arch, best_rmse, loss.cpu().detach().numpy(), time()-search_start))
			controller.print_prob()
			if len(performance.keys()) == SPACE: break

	elif args.mode == 'random':
		search_start = time()
		performance = {}
		best_arch, best_rmse = None, 100000
		args.search_epochs = min(args.search_epochs, SPACE)
		
		archs = []
		for search_epoch in range(args.search_epochs):
			arch = sample_arch() if dim == 2 else sample_arch_triple()
			archs.append(arch)

		for search_epoch in range(args.search_epochs):	
			# if len(performance.keys()) == 200: break
			if search_epoch == 200: break

			# # sample an arch
			# if dim == 2:
			# 	arch = sample_arch()
			# 	# while str(arch) in performance.keys():
			# 	# 	arch = sample_arch()
			# elif dim == 3:
			# 	arch = sample_arch_triple()
			# 	# while str(arch) in performance.keys():
			# 	# 	arch = sample_arch_triple()
			arch = archs[search_epoch]

			# print(next(arch['mlp']['p'].parameters()))

			arch_start = time()
			# rmse = np.random.randn()
			if dim == 2:
				rmse = get_arch_performance(arch, num_users, num_items, train_queue, test_queue, args)
			elif dim == 3:
				rmse = get_arch_performance_triple(arch, num_ps, num_qs, num_rs, train_queue, test_queue, args)
			# performance[str(arch)] = rmse
			if rmse < best_rmse: best_arch, best_rmse = arch, rmse

			op = arch['binary'] if dim == 2 else arch['triple']
			best_op = best_arch['binary'] if dim == 2 else best_arch['triple']

			logging.info('search_epoch: %d finish, arch: %s, rmse: %.4f, arch spent: %.4f' % (
				search_epoch, op, rmse, time()-arch_start))
			logging.info('search_epoch: %d, best_arch: %s, best_rmse: %.4f, time spent: %.4f' % (
				search_epoch, best_op, best_rmse, time()-search_start))

	elif 'darts' in args.mode:
		search_start = time()
		if args.mode == 'binarydarts':
			if dim == 2:
				model = Network(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
			elif dim == 3:
				model = Network_Triple(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'binarydarts_mlp' or args.mode == 'traindarts':
			if dim == 2:
				model = Network_MLP(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
			elif dim == 3:
				model = Network_MLP_Triple(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'darts':
			if dim == 2:
				model = Network_MLP(num_users, num_items, args.embedding_dim, args.weight_decay, prox=1).cuda()
			elif dim == 3:
				model = Network_MLP_Triple(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay, prox=1).cuda()
		elif args.mode == 'proxydarts':
			model = Network_MLP(num_users, num_items, args.embedding_dim, args.weight_decay, prox=2).cuda()

		optimizer = torch.optim.Adagrad(model.parameters(), args.lr)
		arch_optimizer = torch.optim.Adam(model.arch_parameters(), args.arch_lr)

		losses = []
		for train_epoch in range(args.train_epochs):
			if dim == 2:
				g, gp, loss, loss_valid = train_search(train_queue, valid_queue, model, optimizer, arch_optimizer, args)
			elif dim == 3:
				g, gp, loss, loss_valid = train_search_triple(train_queue, valid_queue, model, optimizer, arch_optimizer, args)
			losses.append(loss)

			logging.info('train_epoch: %d, loss: %.4f, loss_valid: %.4f, time spent: %.4f' % (
				train_epoch, loss, loss_valid, time()-search_start))

			logging.info('genotype: %s' % g)  
			print('genotype_p: %s' % gp)
			if train_epoch > 1500:
				if abs(losses[-2]-losses[-1])/losses[-1] < 1e-4/train_queue[0].shape[0] or np.isnan(losses[-1]):
					break
		if args.mode == 'binarydarts_mlp' or args.mode == 'traindarts':
			torch.save(model.state_dict(), '/data3/chenxiangning/models/'+save_name)

	elif args.mode == 'hyperopt':
		start = time()
		from hyperopt import fmin, tpe, hp

		# def get_cfg_performance(cfg):
		# 	if dim == 2:
		# 		arch = {'unary': {'p': cfg['p_unary'], 'q': cfg['q_unary']}, 
		# 				'assist': {'p': cfg['p_assist'], 'q': cfg['q_assist']}, 
		# 				'binary': cfg['binary']}
		# 		rmse = get_arch_performance(arch, num_users, num_items, train_queue, test_queue, args)
		# 	elif dim == 3:
		# 		arch = {'unary': {'p': cfg['p_unary'], 'q': cfg['q_unary'], 'r': cfg['r_unary']}, 
		# 				'assist': {'p': cfg['p_assist'], 'q': cfg['q_assist'], 'r': cfg['r_assist']}, 
		# 				'triple': cfg['triple']}
		# 		rmse = get_arch_performance_triple(arch, num_ps, num_qs, num_rs, train_queue, test_queue, args)
		# 	logging.info('arch: %s, rmse: %.4f, time spent: %.4f' % (
		# 		arch, rmse, time()-start))
		# 	return rmse

		# if dim == 2:
		# 	space = {'p_unary': hp.choice('p_unary', PRIMITIVES_UNARY),
		# 			 'p_assist': hp.choice('p_assist', PRIMITIVES_ASSIST),
		# 			 'q_unary': hp.choice('q_unary', PRIMITIVES_UNARY),
		# 			 'q_assist': hp.choice('q_assist', PRIMITIVES_ASSIST),
		# 			 'binary': hp.choice('binary', PRIMITIVES_BINARY),
		# 			 }
		# elif dim == 3:
		# 	space = {'p_unary': hp.choice('p_unary', PRIMITIVES_UNARY),
		# 			 'p_assist': hp.choice('p_assist', PRIMITIVES_ASSIST),
		# 			 'q_unary': hp.choice('q_unary', PRIMITIVES_UNARY),
		# 			 'q_assist': hp.choice('q_assist', PRIMITIVES_ASSIST),
		# 			 'r_unary': hp.choice('r_unary', PRIMITIVES_UNARY),
		# 			 'r_assist': hp.choice('r_assist', PRIMITIVES_ASSIST),
		# 			 'triple': hp.choice('triple', PRIMITIVES_TRIPLE),
		# 			 }
		# best = fmin(fn=get_cfg_performance,
		# 			space=space,
		# 			algo=tpe.suggest,
		# 			max_evals=200)

		def get_cfg_performance(cfg):
			arch = {}; arch['mlp'] = {}
			if dim == 2:
				arch = sample_arch()
				arch['binary'] = cfg['binary']
				update_arch(arch['mlp']['p'], cfg['mlp_p'])
				update_arch(arch['mlp']['q'], cfg['mlp_q'])
				rmse = get_arch_performance(arch, num_users, num_items, train_queue, test_queue, args)
			elif dim == 3:
				arch = sample_arch_triple()
				arch['triple'] = cfg['triple']
				update_arch(arch['mlp']['p'], cfg['mlp_p'])
				update_arch(arch['mlp']['q'], cfg['mlp_q'])
				update_arch(arch['mlp']['r'], cfg['mlp_r'])
				rmse = get_arch_performance_triple(arch, num_ps, num_qs, num_rs, train_queue, test_queue, args)

			op = arch['binary'] if dim == 2 else arch['triple']
			logging.info('arch: %s, rmse: %.4f, time spent: %.4f' % (
				arch, rmse, time()-start))
			return rmse

		if dim == 2:
			space = {'mlp_p': [hp.uniform('mlp_p%d'%i, -1.0, 1.0) for i in range(25)],
					 'mlp_q': [hp.uniform('mlp_q%d'%i, -1.0, 1.0) for i in range(25)],
					 'binary': hp.choice('binary', PRIMITIVES_BINARY),
					 }
		elif dim == 3:
			space = {'mlp_p': [hp.uniform('mlp_p%d'%i, -1.0, 1.0) for i in range(25)],
					 'mlp_q': [hp.uniform('mlp_q%d'%i, -1.0, 1.0) for i in range(25)],
					 'mlp_r': [hp.uniform('mlp_r%d'%i, -1.0, 1.0) for i in range(25)],
					 'triple': hp.choice('triple', PRIMITIVES_TRIPLE),
					 }
		best = fmin(fn=get_cfg_performance,
					space=space,
					algo=tpe.suggest,
					max_evals=200)

	elif args.mode == 'libfm':
		start = time()
		# if dim == 2:
		# 	from pyfm import pylibfm
		# 	fm = pylibfm.FM(num_factors=args.embedding_dim, num_iter=args.train_epochs, verbose=True, task="regression", 
		# 		initial_learning_rate=args.lr, learning_rate_schedule="optimal")
		# 	fm.fit(train_queue[0], train_queue[1])
		# 	inferences = fm.predict(test_queue[0])
		# 	mse = mean_squared_error(test_queue[1], inferences)
		# 	rmse = np.sqrt(mse)
		# 	logging.info('rmse: %.4f[%.4f]' % (rmse, time()-start))
		
		# elif dim == 3:
		# 	from tffm import TFFMRegressor
		# 	import tensorflow as tf
		# 	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
		# 	model = TFFMRegressor(order=3,
		# 						  rank=args.embedding_dim,
		# 						  optimizer=tf.train.AdagradOptimizer(learning_rate=args.lr),
		# 						  n_epochs=100,
		# 						  # batch_size=1076946,
		# 						  batch_size=4096,
		# 						  init_std=0.001,
		# 						  reg=args.weight_decay,
		# 						  input_type='sparse',
		# 						  log_dir=os.path.join(args.save, save_name),
		# 						  )
		# 	model.fit(train_queue[0], train_queue[1], show_progress=True)
		# 	inferences = model.predict(test_queue[0])
		# 	mse = mean_squared_error(test_queue[1], inferences)
		# 	rmse = np.sqrt(mse)
		# 	logging.info('rmse: %.4f[%.4f]' % (rmse, time()-start))


		from tffm import TFFMRegressor
		import tensorflow as tf
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
		model = TFFMRegressor(order=dim,
							  rank=args.embedding_dim,
							  optimizer=tf.train.AdagradOptimizer(learning_rate=args.lr),
							  n_epochs=args.train_epochs,
							  # batch_size=1076946,
							  batch_size=4096,
							  init_std=0.001,
							  reg=args.weight_decay,
							  input_type='sparse',
							  log_dir=os.path.join(args.save, save_name),
							  )
		model.fit(train_queue[0], train_queue[1], show_progress=True)
		inferences = model.predict(test_queue[0])
		mse = mean_squared_error(test_queue[1], inferences)
		rmse = np.sqrt(mse)
		logging.info('rmse: %.4f[%.4f]' % (rmse, time()-start))

	elif args.mode == 'autoneural':
		start = time()
		if dim == 2:
			model = AutoNeural(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif dim == 3:
			model = AutoNeural_Triple(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay).cuda()
		embedding_oprimizer = torch.optim.Adagrad(model.embedding_parameters(), args.lr)
		mlp_optimizer = torch.optim.Adagrad(model.mlp_parameters(), args.lr)
		losses = []

		for train_epoch in range(args.train_epochs):
			if dim == 2:
				loss = train_single(train_queue, model, embedding_oprimizer, args)

				if train_epoch % 15000 == 0:
					_ = train_single(valid_queue, model, mlp_optimizer, args)

				_ = train_single(valid_queue, model, mlp_optimizer, args)
				rmse = evaluate(model, test_queue)
			elif dim == 3:
				loss = train_single_triple(train_queue, model, embedding_oprimizer, args)
				_ = train_single_triple(valid_queue, model, mlp_optimizer, args)
				rmse = evaluate_triple(model, test_queue)

			losses.append(loss)
			logging.info('train_epoch: %d, loss: %.4f, rmse: %.4f[%.4f]' % (
				train_epoch, loss, rmse, time()-start))

	elif args.mode == 'random_nas':
		search_start = time()
		performance = {}
		best_arch, best_rmse = None, 100000
	
		for search_epoch in range(args.search_epochs):	
			if len(performance.keys()) == 200: break

			arch = sample_arch_nas()
			while str(arch) in performance.keys(): arch = sample_arch_nas()

			arch_start = time()

			if dim == 2:
				rmse = get_arch_performance_nas(arch, num_users, num_items, train_queue, test_queue, args)
			elif dim == 3:
				rmse = get_arch_performance_nas_triple(arch, num_ps, num_qs, num_rs, train_queue, test_queue, args)
			
			performance[str(arch)] = rmse
			if rmse < best_rmse: best_arch, best_rmse = arch, rmse

			logging.info('search_epoch: %d finish, arch: %s, rmse: %.4f, arch spent: %.4f' % (
				search_epoch, arch, rmse, time()-arch_start))
			logging.info('search_epoch: %d, best_arch: %s, best_rmse: %.4f, time spent: %.4f' % (
				search_epoch, best_arch, best_rmse, time()-search_start))
			if len(performance.keys()) == SPACE: break

	elif args.mode == 'reinforce_nas':
		search_start = time()
		performance = {}
		best_arch, best_rmse = None, 100000
		controller = Controller_NAS(args.controller).cuda()

		if args.controller == 'RNN':
			optimizer = torch.optim.Adam(controller.parameters(), args.controller_lr)
		elif args.controller == 'PURE':
			optimizer = torch.optim.Adam(controller.arch_parameters(), args.controller_lr)
		for search_epoch in range(args.search_epochs):
			if len(performance.keys()) == 200: break

			archs = controller.sample_arch(args.controller_batch_size)

			rewards = []
			for arch in archs:
				if str(arch) in performance.keys():
					rmse = performance[str(arch)]
				else:
					arch_start = time()
					if dim == 2:
						rmse = get_arch_performance_nas(arch, num_users, num_items, train_queue, test_queue, args)
					elif dim == 3:
						rmse = get_arch_performance_nas_triple(arch, num_ps, num_qs, num_rs, train_queue, test_queue, args)
					performance[str(arch)] = rmse
					logging.info('search_epoch: %d, arch: %s, rmse: %.4f, arch spent: %.4f' % (
						search_epoch, arch, rmse, time()-arch_start))

				if rmse < best_rmse: best_arch, best_rmse = arch, rmse
				if np.isnan(rmse): 
					rewards.append(0.0)
				else:
					rewards.append(1.0 / (rmse + 1e-5))

			optimizer.zero_grad()
			controller.zero_grad()
			loss = controller.compute_loss(rewards)
			loss.backward()
			optimizer.step()
			logging.info('search_epoch: %d, best_arch: %s, best_rmse: %.4f, loss: %.4f, time_spent: %.4f' % (
				search_epoch, best_arch, best_rmse, loss.cpu().detach().numpy(), time()-search_start))
			controller.print_prob()
			if len(performance.keys()) == SPACE: break

	else:
		start = time()
		if args.mode == 'ncf':
			if dim == 2:
				model = NCF(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
			elif dim == 3:
				model = NCF_Triple(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'deepwide':
			if dim == 2:
				model = DeepWide(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
			elif dim == 3:
				model = DeepWide_Triple(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'altgrad':
			model = AltGrad(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'convncf':
			model = ConvNCF(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'outer':
			model = Outer(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'conv':
			model = Conv(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'plus':
			model = Plus(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'max':
			model = Max(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'min':
			model = Min(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'cp':
			model = CP(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'tucker':
			model = TuckER(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'test':
			if dim == 2:
				arch = {'unary': {'p': 'norm_1', 'q': 'norm_1'}, 'assist': {'p': 'I', 'q': 'I'}, 'binary': 'concat'}
				model = Network_Single(num_users, num_items, args.embedding_dim, arch, args.weight_decay).cuda()
			elif dim == 3:
				arch = {'unary': {'p': 'norm_1', 'q': 'norm_1', 'r': 'norm_1'}, 
					'assist': {'p': 'sign', 'q': 'sign', 'r': 'sign'}, 'triple': 'multiply_multiply'}
				model = Network_Single_Triple(num_ps, num_qs, num_rs, args.embedding_dim, arch, args.weight_decay).cuda()
		elif args.mode == 'test_mlp':
			if dim == 2:
				arch = 'plus'
				model = Network_MLP_Single(num_users, num_items, args.embedding_dim, arch, args.weight_decay, 
					'/data3/chenxiangning/models/traindarts_ml-100k_2_Adagrad0.05_1e-05_1').cuda()
			elif dim == 3:
				arch = '2_plus_multiply'
				model = Network_MLP_Single_Triple(num_ps, num_qs, num_rs, args.embedding_dim, arch, args.weight_decay,
					'/data3/chenxiangning/models/binarydarts_mlp_youtube_small_16_Adagrad0.05_1e-05_1').cuda()

		if args.mode == 'test_mlp':
			optimizer = torch.optim.Adagrad(model.train_parameters(), args.lr)
		else:
			optimizer = torch.optim.Adagrad(model.parameters(), args.lr)
		losses = []

		for train_epoch in range(args.train_epochs):
			if dim == 2:
				if args.minibatch:
					loss = train_single_minibatch(train_queue, model, optimizer, args)
				else:
					loss = train_single(train_queue, model, optimizer, args)
			elif dim == 3:
				if args.minibatch:
					loss = train_single_triple_minibatch(train_queue, model, optimizer, args)
				else:
					loss = train_single_triple(train_queue, model, optimizer, args)
			losses.append(loss)
			if not 'test' in args.mode:
				if train_epoch > 100:
					down = 4096 if args.minibatch else train_queue[0].shape[0]
					if (losses[-2]-losses[-1])/losses[-1] < 1e-4/down or np.isnan(losses[-1]):
						break
			if dim == 2:
				rmse = evaluate(model, test_queue)
				if train_epoch % 50 == 0:
					hr, ndcg = evaluate_hr_ndcg(model, test_queue)
			elif dim == 3:
				rmse = evaluate_triple(model, test_queue)
			if train_epoch % 50 == 0:
				logging.info('train_epoch: %d, loss: %.4f, rmse: %.4f, hr: %.4f, ndcg: %.4f[%.4f]' % (
					train_epoch, loss, rmse, hr, ndcg, time()-start))
			else:
				logging.info('train_epoch: %d, loss: %.4f, rmse: %.4f[%.4f]' % (
					train_epoch, loss, rmse, time()-start))
			# break
	print(save_name)



	
	





















