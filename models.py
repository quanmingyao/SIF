import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math

PRIMITIVES_UNARY = ['norm_0.5', 'norm_1', 'norm_2']
PRIMITIVES_ASSIST = ['I', '-I', 'sign']

# PRIMITIVES_UNARY = ['norm_1']
# PRIMITIVES_ASSIST = ['sign']

PRIMITIVES_BINARY = ['plus', 'multiply', 'max', 'min', 'concat']

PRIMITIVES_TRIPLE = ['0_plus_multiply', '0_plus_max', '0_plus_min', '0_plus_concat',
					 '0_multiply_plus', '0_multiply_max', '0_multiply_min', '0_multiply_concat',
					 '0_max_plus', '0_max_multiply', '0_max_min', '0_max_concat',
					 '0_min_plus', '0_min_multiply', '0_min_max', '0_min_concat',
					 '1_plus_multiply', '1_plus_max', '1_plus_min', '1_plus_concat',
					 '1_multiply_plus', '1_multiply_max', '1_multiply_min', '1_multiply_concat',
					 '1_max_plus', '1_max_multiply', '1_max_min', '1_max_concat',
					 '1_min_plus', '1_min_multiply', '1_min_max', '1_min_concat',
					 '2_plus_multiply', '2_plus_max', '2_plus_min', '2_plus_concat',
					 '2_multiply_plus', '2_multiply_max', '2_multiply_min', '2_multiply_concat',
					 '2_max_plus', '2_max_multiply', '2_max_min', '2_max_concat',
					 '2_min_plus', '2_min_multiply', '2_min_max', '2_min_concat',
					 'plus_plus', 'multiply_multiply', 'max_max', 'min_min', 'concat_concat',
					 ]

PRIMITIVES_NAS = [0, 2, 4, 8, 16]

SPACE = len(PRIMITIVES_BINARY) * np.square(len(PRIMITIVES_UNARY)) * \
	np.square(len(PRIMITIVES_ASSIST))

SPACE_NAS = pow(len(PRIMITIVES_NAS), 5)

OPS = {
	'plus': lambda p, q: p + q,
	'multiply': lambda p, q: p * q,
	'max': lambda p, q: torch.max(torch.stack((p, q)), dim=0)[0],
	'min': lambda p, q: torch.min(torch.stack((p, q)), dim=0)[0],
	'concat': lambda p, q: torch.cat([p, q], dim=-1),
	'norm_0': lambda p: torch.ones_like(p),
	'norm_0.5': lambda p: torch.sqrt(torch.abs(p) + 1e-7),
	'norm_1': lambda p: torch.abs(p),
	'norm_2': lambda p: p ** 2,
	'I': lambda p: torch.ones_like(p),
	'-I': lambda p: -torch.ones_like(p),
	'sign': lambda p: torch.sign(p),
}

def ops_triple(triple, p, q, r):
	if triple == 'plus_plus':
		return OPS['plus'](OPS['plus'](p, q), r)
	elif triple == 'multiply_multiply':
		return OPS['plus'](OPS['plus'](p, q), r)
	elif triple == 'max_max':
		return OPS['max'](OPS['max'](p, q), r)
	elif triple == 'min_min':
		return OPS['min'](OPS['min'](p, q), r)
	elif triple == 'concat_concat':
		return OPS['concat'](OPS['concat'](p, q), r)
	else:
		ops = triple.split('_')
		if ops[0] == '0':
			return OPS[ops[2]](OPS[ops[1]](p, q), r)
		elif ops[0] == '1':
			return OPS[ops[2]](OPS[ops[1]](p, r), q)
		elif ops[0] == '2':
			return OPS[ops[2]](OPS[ops[1]](q, r), p)

def _concat(xs):
	return torch.cat([x.view(-1) for x in xs])

def MixedUnary(embedding, weights):
	return torch.sum(torch.stack([w * OPS[primitive](embedding) \
		for w,primitive in zip(weights,PRIMITIVES_UNARY)]), 0)

def MixedAssist(embedding, weights):
	return torch.sum(torch.stack([w * OPS[primitive](embedding) \
		for w,primitive in zip(weights,PRIMITIVES_ASSIST)]), 0)

def MixedBinary(embedding_p, embedding_q, weights, FC):
	# return torch.sum(torch.stack([w * OPS[primitive](embedding_p, embedding_q) \
		# for w,primitive in zip(weights,PRIMITIVES_BINARY)]), 0)
	return torch.sum(torch.stack([w * fc(OPS[primitive](embedding_p, embedding_q)) \
		for w,primitive,fc in zip(weights,PRIMITIVES_BINARY,FC)]), 0)

def MixedTriple(embedding_p, embedding_q, embedding_r, weights, FC):
	return torch.sum(torch.stack([w * fc(ops_triple(primitive, embedding_p, embedding_q, embedding_r)) \
		for w,primitive,fc in zip(weights,PRIMITIVES_TRIPLE,FC)]), 0)

def initialize_alpha(length):
	alpha = torch.ones(length, dtype=torch.float, device='cuda') / length
	alpha += torch.randn(length, device='cuda') * 1e-3
	return Variable(alpha, requires_grad=True)

def copy(x, y, with_grad=True, triple=False):
	if with_grad:
		x['unary']['p'] = y['unary']['p'].clone()
		x['unary']['q'] = y['unary']['q'].clone()
		x['assist']['p'] = y['assist']['p'].clone()
		x['assist']['q'] = y['assist']['q'].clone()
		if triple:
			x['triple'] = y['triple'].clone()
			x['unary']['r'] = y['unary']['r'].clone()
			x['assist']['r'] = y['assist']['r'].clone()
		else:
			x['binary'] = y['binary'].clone()
	else:
		x['unary']['p'].data = y['unary']['p'].data.clone()
		x['unary']['q'].data = y['unary']['q'].data.clone()
		x['assist']['p'].data = y['assist']['p'].data.clone()
		x['assist']['q'].data = y['assist']['q'].data.clone()
		if triple:
			x['triple'].data = y['triple'].data.clone()
			x['unary']['r'].data = y['unary']['r'].data.clone()
			x['assist']['r'].data = y['assist']['r'].data.clone()
		else:
			x['binary'].data = y['binary'].data.clone()

def binarize(x):
	m = torch.max(x)
	x.data[x<m] = 0.0
	x.data[x==m] = 1.0
	return x

def constrain(p):
	c = torch.norm(p, p=2, dim=1, keepdim=True)
	c[c<1] = 1.0
	p.data.div_(c)


class Virtue(nn.Module):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(Virtue, self).__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.embedding_dim = embedding_dim
		self.reg = reg
		self._UsersEmbedding = nn.Embedding(num_users, embedding_dim)
		self._ItemsEmbedding = nn.Embedding(num_items, embedding_dim)

	def compute_loss(self, inferences, labels, regs):
		labels = torch.reshape(labels, [-1,1])
		loss = F.mse_loss(inferences, labels)
		return loss + regs

class NCF(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(NCF, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)
		self._W = nn.Linear(2*embedding_dim, embedding_dim)
		# self._FC = nn.Sequential(
		# 	nn.Linear(embedding_dim, embedding_dim),
		# 	nn.Tanh(),
		# 	nn.Linear(embedding_dim, 1, bias=False))

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))
		constrain(next(self._W.parameters()))

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		gmf_out = users_embedding * items_embedding
		mlp_out = self._W(torch.cat([users_embedding, items_embedding], dim=-1))
		inferences = self._FC(F.tanh(gmf_out + mlp_out))
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs

class DeepWide(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(DeepWide, self).__init__(num_users, num_items, embedding_dim, reg)
		# self._FC = nn.Linear(2*embedding_dim, 1, bias=False)
		self._FC = nn.Sequential(
			nn.Linear(2*embedding_dim, embedding_dim),
			nn.ReLU(),
			nn.Linear(embedding_dim, 1, bias=False))

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = self._FC(torch.cat([users_embedding, items_embedding], dim=-1))
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs

class AltGrad(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(AltGrad, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = self._FC(users_embedding * items_embedding)
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs

class ConvNCF(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(ConvNCF, self).__init__(num_users, num_items, embedding_dim, reg)
		self.num_conv = int(math.log(embedding_dim, 2))
		self._Conv = []
		for i in range(self.num_conv-1):
			self._Conv.append(nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1))
			self._Conv.append(nn.ReLU())
		self._Conv.append(nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1))
		self._Conv = nn.Sequential(*self._Conv)

	def forward(self, users, items):

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		outer_result = torch.bmm(users_embedding.view(-1,self.embedding_dim,1), 
			items_embedding.view(-1,1,self.embedding_dim))

		outer_result = torch.unsqueeze(outer_result, 1)

		inferences = self._Conv(outer_result).view(-1, 1)
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs

class Plus(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(Plus, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = self._FC(users_embedding + items_embedding)
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs

class Max(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(Max, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = self._FC(OPS['max'](users_embedding, items_embedding))
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs

class Min(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(Min, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = self._FC(OPS['min'](users_embedding, items_embedding))
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs

class Conv(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(Conv, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = []
		for i in range(self.embedding_dim):
			tmp = torch.zeros(users_embedding.size(0), 1, dtype=torch.float, device='cuda', requires_grad=False)
			for j in range(i+1):
				tmp += torch.reshape(users_embedding[:,j]*items_embedding[:,i-j], [-1,1])
			inferences.append(tmp)
		inferences = torch.cat(inferences, -1)
		inferences = self._FC(inferences)
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs

class Outer(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(Outer, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim**2, 1, bias=False)

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = torch.bmm(users_embedding.view(-1,self.embedding_dim,1), 
			items_embedding.view(-1,1,self.embedding_dim)).view(-1,self.embedding_dim**2)
		inferences = self._FC(inferences)
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs

# class Network_Single(Virtue):

# 	def __init__(self, num_users, num_items, embedding_dim, arch, reg):
# 		super(Network_Single, self).__init__(num_users, num_items, embedding_dim, reg)
# 		self.arch = arch
# 		if arch['binary'] == 'concat':
# 			self._FC = nn.Linear(2*embedding_dim, 1, bias=False)
# 		else:
# 			self._FC = nn.Linear(embedding_dim, 1, bias=False)

# 	def forward(self, users, items):
# 		constrain(next(self._FC.parameters()))

# 		users_embedding = self._UsersEmbedding(users)
# 		items_embedding = self._ItemsEmbedding(items)

# 		users_embedding_trans = OPS[self.arch['unary']['p']](users_embedding) * \
# 			OPS[self.arch['assist']['p']](users_embedding)
# 		items_embedding_trans = OPS[self.arch['unary']['q']](items_embedding) * \
# 			OPS[self.arch['assist']['q']](items_embedding)

# 		inferences = self._FC(OPS[self.arch['binary']](users_embedding_trans, 
# 			items_embedding_trans))
# 		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))

# 		return inferences, regs

class Network_Single(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, arch, reg):
		super(Network_Single, self).__init__(num_users, num_items, embedding_dim, reg)
		self.arch = arch
		if arch['binary'] == 'concat':
			self._FC = nn.Linear(2*embedding_dim, 1, bias=False)
		else:
			self._FC = nn.Linear(embedding_dim, 1, bias=False)

		# print(next(self.arch['mlp']['p'].parameters()))

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))
		for i,p in enumerate(self.arch['mlp']['p'].parameters()):
			if len(p.size()) == 1: continue
			constrain(list(self.arch['mlp']['p'].parameters())[i])
		for i,q in enumerate(self.arch['mlp']['q'].parameters()):
			if len(p.size()) == 1: continue
			constrain(list(self.arch['mlp']['q'].parameters())[i])

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		users_embedding_trans = self.arch['mlp']['p'](users_embedding.view(-1,1)).view(users_embedding.size())
		items_embedding_trans = self.arch['mlp']['q'](items_embedding.view(-1,1)).view(items_embedding.size())

		inferences = self._FC(OPS[self.arch['binary']](users_embedding_trans, 
			items_embedding_trans))
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs

class Network(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg, prox=3):
		super(Network, self).__init__(num_users, num_items, embedding_dim, reg)

		self._FC = nn.ModuleList()
		for primitive in PRIMITIVES_BINARY:
			if primitive == 'concat':
				self._FC.append(nn.Linear(2*embedding_dim, 1, bias=False))
			else:
				self._FC.append(nn.Linear(embedding_dim, 1, bias=False))
		self.prox = prox
		self._initialize_alphas()

	def _initialize_alphas(self):
		self._arch_parameters = {}
		self._arch_parameters['unary'], self._arch_parameters['assist'] = {}, {}
		self._arch_parameters['unary']['p'] = initialize_alpha(len(PRIMITIVES_UNARY))
		self._arch_parameters['unary']['q'] = initialize_alpha(len(PRIMITIVES_UNARY))
		self._arch_parameters['assist']['p'] = initialize_alpha(len(PRIMITIVES_ASSIST))
		self._arch_parameters['assist']['q'] = initialize_alpha(len(PRIMITIVES_ASSIST))
		self._arch_parameters['binary'] = initialize_alpha(len(PRIMITIVES_BINARY))

	def arch_parameters(self):
		return [self._arch_parameters['unary']['p'],
				self._arch_parameters['unary']['q'],
				self._arch_parameters['assist']['p'],
				self._arch_parameters['assist']['q'],
				self._arch_parameters['binary']]

	def new(self):
		model_new = Network(self.num_users, self.num_items, self.embedding_dim, self.reg).cuda()
		for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
			x.data = y.data.clone()
			try:
				x.grad = y.grad.clone()
			except:
				pass
		return model_new

	def _binarize(self):
		self._cache = {}
		self._cache['unary'], self._cache['assist'] = {}, {}
		copy(self._cache, self._arch_parameters)
		self._arch_parameters['unary']['p'] = binarize(self._arch_parameters['unary']['p'])
		self._arch_parameters['unary']['q'] = binarize(self._arch_parameters['unary']['q'])
		self._arch_parameters['assist']['p'] = binarize(self._arch_parameters['assist']['p'])
		self._arch_parameters['assist']['q'] = binarize(self._arch_parameters['assist']['q'])
		self._arch_parameters['binary'] = binarize(self._arch_parameters['binary'])

	def _recover(self):
		copy(self._arch_parameters, self._cache, False)

	def forward(self, users, items):
		if self.prox == 2:
			self._binarize()
		for i in range(len(PRIMITIVES_BINARY)):
			constrain(next(self._FC[i].parameters()))

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		users_embedding_trans = MixedUnary(users_embedding, 
			self._arch_parameters['unary']['p']) * MixedAssist(users_embedding, 
			F.softmax(self._arch_parameters['assist']['p'], dim=-1))
		items_embedding_trans = MixedUnary(items_embedding, 
			self._arch_parameters['unary']['q']) * MixedAssist(items_embedding,
			F.softmax(self._arch_parameters['assist']['q'], dim=-1))

		inferences = MixedBinary(users_embedding_trans, items_embedding_trans,
			self._arch_parameters['binary'], self._FC)

		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))

		return inferences, regs

	def genotype(self):
		genotype = {}
		genotype['unary'], genotype['assist'] = {}, {}
		genotype['unary']['p'] = PRIMITIVES_UNARY[self._arch_parameters['unary']['p'].argmax().cpu().numpy()]
		genotype['unary']['q'] = PRIMITIVES_UNARY[self._arch_parameters['unary']['q'].argmax().cpu().numpy()]
		genotype['assist']['p'] = PRIMITIVES_ASSIST[self._arch_parameters['assist']['p'].argmax().cpu().numpy()]
		genotype['assist']['q'] = PRIMITIVES_ASSIST[self._arch_parameters['assist']['q'].argmax().cpu().numpy()]
		genotype['binary'] = PRIMITIVES_BINARY[self._arch_parameters['binary'].argmax().cpu().numpy()]

		genotype_p = {}
		genotype_p['unary'], genotype_p['assist'] = {}, {}
		genotype_p['unary']['p'] = F.softmax(self._arch_parameters['unary']['p'], dim=-1)
		genotype_p['unary']['q'] = F.softmax(self._arch_parameters['unary']['q'], dim=-1)
		genotype_p['assist']['p'] = F.softmax(self._arch_parameters['assist']['p'], dim=-1)
		genotype_p['assist']['q'] = F.softmax(self._arch_parameters['assist']['q'], dim=-1)
		genotype_p['binary'] = F.softmax(self._arch_parameters['binary'], dim=-1)
		return genotype, genotype_p

	def step(self, users_train, items_train, labels_train, users_valid, 
		items_valid, labels_valid, lr, arch_optimizer, unrolled):

		self.train()
		self.zero_grad()
		arch_optimizer.zero_grad()
		if unrolled:
			loss = self._backward_step_unrolled(users_train, items_train, labels_train,
				users_valid, items_valid, labels_valid, lr)
		else:
			loss = self._backward_step(users_valid, items_valid, labels_valid)
		arch_optimizer.step()
		return loss

	def _backward_step(self, users_valid, items_valid, labels_valid):
		if self.prox == 3:
			self._binarize()
		inferences, regs = self(users_valid, items_valid)
		loss = self.compute_loss(inferences, labels_valid, regs)
		loss.backward()
		if self.prox == 3:
			self._recover()
		return loss

	def _backward_step_unrolled(self, users_train, items_train, labels_train,
		users_valid, items_valid, labels_valid, lr):
		if self.prox == 3:
			self._binarize()
		unrolled_model = self._compute_unrolled_model(
			users_train, items_train, labels_train, lr)
		unrolled_inference, unrolled_regs = unrolled_model(users_valid, items_valid)
		unrolled_loss = unrolled_model.compute_loss(unrolled_inference, labels_valid, unrolled_regs)

		unrolled_loss.backward()
		dalpha = [v.grad for v in unrolled_model.arch_parameters()]
		vector = [v.grad for v in unrolled_model.parameters()]
		implicit_grads = self._hessian_vector_product(vector, users_train, items_train, labels_train)

		for g,ig in zip(dalpha,implicit_grads):
			g.sub_(lr, ig)

		for v,g in zip(self.arch_parameters(), dalpha):
			v.grad = g.clone()
		if self.prox == 3:
			self._recover()
		return unrolled_loss

	def _compute_unrolled_model(self, users_train, items_train, labels_train, lr):
		inferences, regs = self(users_train, items_train)
		loss = self.compute_loss(inferences, labels_train, regs)
		theta = _concat(self.parameters())
		dtheta = _concat(torch.autograd.grad(loss, self.parameters())) + \
			self.reg * theta
		unrolled_model = self._construct_model_from_theta(
			theta.sub(lr, dtheta))
		return unrolled_model

	def _construct_model_from_theta(self, theta):
		model_new = self.new()
		model_dict = self.state_dict()
		params, offset = {}, 0
		for k,v in self.named_parameters():
			v_length = np.prod(v.size())
			params[k] = theta[offset: offset+v_length].view(v.size())
			offset += v_length

		assert offset == len(theta)
		model_dict.update(params)
		model_new.load_state_dict(model_dict)
		return model_new.cuda()

	def _hessian_vector_product(self, vector, users, items, labels, r=1e-2):
		R = r / _concat(vector).norm()
		for p,v in zip(self.parameters(), vector):
			p.data.add_(R, v)
		inferences, regs = self(users, items)
		loss = self.compute_loss(inferences, labels, regs)
		grads_p = torch.autograd.grad(loss, self.arch_parameters())

		for p,v in zip(self.parameters(), vector):
			p.data.sub_(2*R, v)
		inferences, regs = self(users, items)
		loss = self.compute_loss(inferences, labels, regs)
		grads_n = torch.autograd.grad(loss, self.arch_parameters())

		for p,v in zip(self.parameters(), vector):
			p.data.add_(R, v)

		return [(x-y).div_(2*R) for x,y in zip(grads_p,grads_n)]

class Network_MLP(Network):

	def __init__(self, num_users, num_items, embedding_dim, reg, prox=3):
		super(Network_MLP, self).__init__(num_users, num_items, embedding_dim, reg, prox)

	def _initialize_alphas(self):
		self.mlp_p = nn.Sequential(
			nn.Linear(1, 8),
			nn.Tanh(),
			nn.Linear(8, 1)).cuda()
		self.mlp_q = nn.Sequential(
			nn.Linear(1, 8),
			nn.Tanh(),
			nn.Linear(8, 1)).cuda()
		self._arch_parameters = {}; self._arch_parameters['mlp'] = {}
		self._arch_parameters['mlp']['p'] = self.mlp_p
		self._arch_parameters['mlp']['q'] = self.mlp_q
		self._arch_parameters['binary'] = initialize_alpha(len(PRIMITIVES_BINARY))

	def arch_parameters(self):
		return list(self._arch_parameters['mlp']['p'].parameters()) + \
			   list(self._arch_parameters['mlp']['q'].parameters()) + \
			   [self._arch_parameters['binary']]

	def new(self):
		model_new = Network_MLP(self.num_users, self.num_items, self.embedding_dim, self.reg, self.prox).cuda()
		for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
			x.data = y.data.clone()
			try:
				x.grad = y.grad.clone()
			except:
				pass
		return model_new

	def _binarize(self):
		self._cache = self._arch_parameters['binary'].clone()
		self._arch_parameters['binary'] = binarize(self._arch_parameters['binary'])

	def _recover(self):
		self._arch_parameters['binary'].data = self._cache.data.clone()

	def forward(self, users, items):
		if self.prox == 2:
			self._binarize()
		for i in range(len(PRIMITIVES_BINARY)):
			constrain(next(self._FC[i].parameters()))
		for i in range(len(self.arch_parameters())):
			if i == len(self.arch_parameters()) - 1: break
			if len(self.arch_parameters()[i].size()) == 1: continue
			constrain(self.arch_parameters()[i])

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		users_embedding_trans = self._arch_parameters['mlp']['p'](users_embedding.view(-1,1)).view(users_embedding.size())
		items_embedding_trans = self._arch_parameters['mlp']['q'](items_embedding.view(-1,1)).view(items_embedding.size())

		if self.prox == 1:
			prob = F.softmax(self._arch_parameters['binary'], dim=-1)
		else:
			prob = self._arch_parameters['binary']

		inferences = MixedBinary(users_embedding_trans, items_embedding_trans,
			prob, self._FC)

		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs

	def genotype(self):
		genotype = PRIMITIVES_BINARY[self._arch_parameters['binary'].argmax().cpu().numpy()]
		if self.prox == 2:
			genotype_p = self._arch_parameters['binary']
		else:
			genotype_p = F.softmax(self._arch_parameters['binary'], dim=-1)
		return genotype, genotype_p

class Network_MLP_Single(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, arch, reg, path):
		super(Network_MLP_Single, self).__init__(num_users, num_items, embedding_dim, reg)
		self.arch = arch
		
		network_mlp = Network_MLP(num_users, num_items, embedding_dim, reg)
		network_mlp.load_state_dict(torch.load(path))
		self.mlp_p = network_mlp.mlp_p
		self.mlp_q = network_mlp.mlp_q
		if arch == 'concat':
			self._FC = nn.Linear(2*embedding_dim, 1, bias=False)
		else:
			self._FC = nn.Linear(embedding_dim, 1, bias=False)
		self._arch_parameters = list(self.mlp_p.parameters()) + list(self.mlp_q.parameters())

	def train_parameters(self):
		return list(self._UsersEmbedding.parameters()) + list(self._ItemsEmbedding.parameters()) + \
			list(self._FC.parameters())

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))

		for i in range(len(self._arch_parameters)):
			if len(self._arch_parameters[i].size()) == 1: continue
			constrain(self._arch_parameters[i])

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		users_embedding_trans = self.mlp_p(users_embedding.view(-1,1)).view(users_embedding.size())
		items_embedding_trans = self.mlp_q(items_embedding.view(-1,1)).view(items_embedding.size())

		inferences = self._FC(OPS[self.arch](users_embedding_trans, 
			items_embedding_trans))
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs

class AutoNeural(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(AutoNeural, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Sequential(
			nn.Linear(2*embedding_dim, 2*embedding_dim),
			nn.Sigmoid(),
			nn.Linear(2*embedding_dim, 1))

	def forward(self, users, items):
		for p in self._FC.parameters():
			if len(p.size()) == 1: continue
			constrain(p)

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = self._FC(torch.cat([users_embedding,items_embedding], dim=-1))
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))

		return inferences, regs

	def embedding_parameters(self):
		return list(self._UsersEmbedding.parameters()) + list(self._ItemsEmbedding.parameters())

	def mlp_parameters(self):
		return self._FC.parameters()

class NAS(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, arch, reg):
		super(NAS, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = []

		for i in range(len(arch)):
			if i == 0:
				self._FC.append(nn.Linear(2*embedding_dim, int(arch[i])))
			else:
				self._FC.append(nn.Linear(int(arch[i-1]), int(arch[i])))
			self._FC.append(nn.ReLU())
		if len(self._FC) == 0:
			self._FC.append(nn.Linear(2*embedding_dim, 1, bias=False))
		else:
			self._FC.append(nn.Linear(arch[-1], 1, bias=False))
		self._FC = nn.Sequential(*self._FC)

	def forward(self, users, items):
		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = self._FC(torch.cat([users_embedding, items_embedding], dim=-1))
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs

class Virtue_Triple(nn.Module):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(Virtue_Triple, self).__init__()
		self.num_ps = num_ps
		self.num_qs = num_qs
		self.num_rs = num_rs
		self.embedding_dim = embedding_dim
		self.reg = reg
		self._PsEmbedding = nn.Embedding(num_ps, embedding_dim)
		self._QsEmbedding = nn.Embedding(num_qs, embedding_dim)
		self._RsEmbedding = nn.Embedding(num_rs, embedding_dim)

	def compute_loss(self, inferences, labels, regs):
		labels = torch.reshape(labels, [-1,1])
		loss = F.mse_loss(inferences, labels)
		return loss + regs
	
class NCF_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(NCF_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)
		self._W = nn.Linear(3*embedding_dim, embedding_dim)

	def forward(self, ps, qs, rs):
		constrain(next(self._FC.parameters()))
		constrain(next(self._W.parameters()))

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		gmf_out = ps_embedding * qs_embedding * rs_embedding
		mlp_out = self._W(torch.cat([ps_embedding, qs_embedding, rs_embedding], dim=-1))
		inferences = self._FC(F.relu(gmf_out + mlp_out))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs

class DeepWide_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(DeepWide_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = nn.Linear(3*embedding_dim, 1, bias=False)

	def forward(self, ps, qs, rs):
		constrain(next(self._FC.parameters()))

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		inferences = self._FC(torch.cat([ps_embedding, qs_embedding, rs_embedding], dim=-1))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs

class CP(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(CP, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def forward(self, ps, qs, rs):
		constrain(next(self._FC.parameters()))

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		inferences = self._FC(ps_embedding * qs_embedding * rs_embedding)
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs

class TuckER(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(TuckER, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		w = torch.empty(embedding_dim, embedding_dim, embedding_dim)
		nn.init.xavier_uniform_(w)
		self._W = torch.nn.Parameter(torch.tensor(w, dtype=torch.float, device='cuda', requires_grad=True))

	def forward(self, ps, qs, rs):
		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		W_after_p = torch.mm(ps_embedding, self._W.view(ps_embedding.size(1), -1))
		W_after_p = W_after_p.view(-1, rs_embedding.size(1), qs_embedding.size(1))
		W_after_r = torch.bmm(rs_embedding.view(-1,1,rs_embedding.size(1)), W_after_p)
		W_after_q = torch.bmm(W_after_r, qs_embedding.view(-1,qs_embedding.size(1),1))
		inferences = W_after_q.view(-1,1)
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs

class Network_Single_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, arch, reg):
		super(Network_Single_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self.arch = arch
		if self.arch['triple'] == 'concat_concat':
			self._FC = nn.Linear(3*embedding_dim, 1, bias=False)
		elif 'concat' in self.arch['triple']:
			self._FC = nn.Linear(2*embedding_dim, 1, bias=False)
		else:
			self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def forward(self, ps, qs, rs):
		constrain(next(self._FC.parameters()))
		for i,p in enumerate(self.arch['mlp']['p'].parameters()):
			if len(p.size()) == 1: continue
			constrain(list(self.arch['mlp']['p'].parameters())[i])
		for i,q in enumerate(self.arch['mlp']['q'].parameters()):
			if len(p.size()) == 1: continue
			constrain(list(self.arch['mlp']['q'].parameters())[i])
		for i,r in enumerate(self.arch['mlp']['r'].parameters()):
			if len(p.size()) == 1: continue
			constrain(list(self.arch['mlp']['r'].parameters())[i])

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(ps)
		rs_embedding = self._RsEmbedding(rs)

		ps_embedding_trans = self.arch['mlp']['p'](ps_embedding.view(-1,1)).view(ps_embedding.size())
		qs_embedding_trans = self.arch['mlp']['q'](qs_embedding.view(-1,1)).view(qs_embedding.size())
		rs_embedding_trans = self.arch['mlp']['r'](rs_embedding.view(-1,1)).view(rs_embedding.size())

		inferences = self._FC(ops_triple(self.arch['triple'], ps_embedding_trans, 
			qs_embedding_trans, rs_embedding_trans))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + \
			torch.norm(rs_embedding))
		return inferences, regs

class Network_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg, prox=3):
		super(Network_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = nn.ModuleList()
		for primitive in PRIMITIVES_TRIPLE:
			if primitive == 'concat_concat':
				self._FC.append(nn.Linear(3*embedding_dim, 1, bias=False))
			elif 'concat' in primitive:
				self._FC.append(nn.Linear(2*embedding_dim, 1, bias=False))
			else:
				self._FC.append(nn.Linear(embedding_dim, 1, bias=False))
		self.prox = prox
		self._initialize_alphas()

	def _initialize_alphas(self):
		self._arch_parameters = {}
		self._arch_parameters['unary'], self._arch_parameters['assist'] = {}, {}
		self._arch_parameters['unary']['p'] = initialize_alpha(len(PRIMITIVES_UNARY))
		self._arch_parameters['unary']['q'] = initialize_alpha(len(PRIMITIVES_UNARY))
		self._arch_parameters['unary']['r'] = initialize_alpha(len(PRIMITIVES_UNARY))
		self._arch_parameters['assist']['p'] = initialize_alpha(len(PRIMITIVES_ASSIST))
		self._arch_parameters['assist']['q'] = initialize_alpha(len(PRIMITIVES_ASSIST))
		self._arch_parameters['assist']['r'] = initialize_alpha(len(PRIMITIVES_ASSIST))
		self._arch_parameters['triple'] = initialize_alpha(len(PRIMITIVES_TRIPLE))

	def arch_parameters(self):
		return [self._arch_parameters['unary']['p'],
				self._arch_parameters['unary']['q'],
				self._arch_parameters['unary']['r'],
				self._arch_parameters['assist']['p'],
				self._arch_parameters['assist']['q'],
				self._arch_parameters['assist']['r'],
				self._arch_parameters['triple']]

	def new(self):
		model_new = Network_Triple(self.num_ps, self.num_qs, self.num_rs, self.embedding_dim, self.reg, self.prox).cuda()
		for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
			x.data = y.data.clone()
			try:
				x.grad = y.grad.clone()
			except:
				pass
		return model_new

	def _binarize(self):
		self._cache = {}
		self._cache['unary'], self._cache['assist'] = {}, {}
		copy(self._cache, self._arch_parameters, triple=True)
		self._arch_parameters['unary']['p'] = binarize(self._arch_parameters['unary']['p'])
		self._arch_parameters['unary']['q'] = binarize(self._arch_parameters['unary']['q'])
		self._arch_parameters['unary']['r'] = binarize(self._arch_parameters['unary']['r'])
		self._arch_parameters['assist']['p'] = binarize(self._arch_parameters['assist']['p'])
		self._arch_parameters['assist']['q'] = binarize(self._arch_parameters['assist']['q'])
		self._arch_parameters['assist']['r'] = binarize(self._arch_parameters['assist']['r'])
		self._arch_parameters['triple'] = binarize(self._arch_parameters['triple'])

	def _recover(self):
		copy(self._arch_parameters, self._cache, with_grad=False, triple=True)

	def forward(self, p, q, r):
		for i in range(len(self._FC)):
			constrain(next(self._FC[i].parameters()))

		ps_embedding = self._PsEmbedding(p)
		qs_embedding = self._QsEmbedding(q)
		rs_embedding = self._RsEmbedding(r)

		ps_embedding_trans = MixedUnary(ps_embedding, 
			self._arch_parameters['unary']['p']) * MixedAssist(ps_embedding, 
			self._arch_parameters['assist']['p'])
		qs_embedding_trans = MixedUnary(qs_embedding, 
			self._arch_parameters['unary']['q']) * MixedAssist(qs_embedding, 
			self._arch_parameters['assist']['q'])
		rs_embedding_trans = MixedUnary(rs_embedding, 
			self._arch_parameters['unary']['r']) * MixedAssist(rs_embedding, 
			self._arch_parameters['assist']['r'])

		inferences = MixedTriple(ps_embedding_trans, qs_embedding_trans, rs_embedding_trans,
			F.softmax(self._arch_parameters['triple'], dim=-1), self._FC)
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + 
			torch.norm(rs_embedding))

		return inferences, regs

	def genotype(self):
		genotype = {}
		genotype['unary'], genotype['assist'] = {}, {}
		genotype['unary']['p'] = PRIMITIVES_UNARY[self._arch_parameters['unary']['p'].argmax().cpu().numpy()]
		genotype['unary']['q'] = PRIMITIVES_UNARY[self._arch_parameters['unary']['q'].argmax().cpu().numpy()]
		genotype['unary']['r'] = PRIMITIVES_UNARY[self._arch_parameters['unary']['r'].argmax().cpu().numpy()]
		genotype['assist']['p'] = PRIMITIVES_ASSIST[self._arch_parameters['assist']['p'].argmax().cpu().numpy()]
		genotype['assist']['q'] = PRIMITIVES_ASSIST[self._arch_parameters['assist']['q'].argmax().cpu().numpy()]
		genotype['assist']['r'] = PRIMITIVES_ASSIST[self._arch_parameters['assist']['r'].argmax().cpu().numpy()]
		genotype['triple'] = PRIMITIVES_TRIPLE[self._arch_parameters['triple'].argmax().cpu().numpy()]

		genotype_p = {}
		genotype_p['unary'], genotype_p['assist'] = {}, {}
		genotype_p['unary']['p'] = F.softmax(self._arch_parameters['unary']['p'], dim=-1)
		genotype_p['unary']['q'] = F.softmax(self._arch_parameters['unary']['q'], dim=-1)
		genotype_p['unary']['r'] = F.softmax(self._arch_parameters['unary']['r'], dim=-1)
		genotype_p['assist']['p'] = F.softmax(self._arch_parameters['assist']['p'], dim=-1)
		genotype_p['assist']['q'] = F.softmax(self._arch_parameters['assist']['q'], dim=-1)
		genotype_p['assist']['r'] = F.softmax(self._arch_parameters['assist']['r'], dim=-1)
		genotype_p['triple'] = F.softmax(self._arch_parameters['triple'], dim=-1)
		return genotype, genotype_p

	def step(self, p_train, q_train, r_train, labels_train, p_valid, q_valid, 
		r_valid, labels_valid, lr, arch_optimizer, unrolled):
		self.train()
		self.zero_grad()
		arch_optimizer.zero_grad()
		if unrolled:
			loss = self._backward_step_unrolled(p_train, q_train, r_train, 
				labels_train, p_valid, q_valid, r_valid, labels_valid, lr)
		else:
			loss = self._backward_step(p_valid, q_valid, r_valid, labels_valid)
		arch_optimizer.step()
		return loss

	def _backward_step(self, p_valid, q_valid, r_valid, labels_valid):
		if self.prox == 3:
			self._binarize()
		inferences, regs = self(p_valid, q_valid, r_valid)
		loss = self.compute_loss(inferences, labels_valid, regs)
		loss.backward()
		if self.prox == 3:
			self._recover()
		return loss

	def _backward_step_unrolled(self, p_train, q_train, r_train, labels_train, 
		p_valid, q_valid, r_valid, labels_valid, lr):
		if self.prox == 3:
			self._binarize()
		unrolled_model = self._compute_unrolled_model(
			p_train, q_train, r_train, labels_train, lr)
		unrolled_inference, unrolled_regs = unrolled_model(p_valid, q_valid, r_valid)
		unrolled_loss = unrolled_model.compute_loss(unrolled_inference, labels_valid, unrolled_regs)

		unrolled_loss.backward()
		dalpha = [v.grad for v in unrolled_model.arch_parameters()]
		vector = [v.grad for v in unrolled_model.parameters()]
		implicit_grads = self._hessian_vector_product(vector, p_train, q_train, r_train, labels_train)

		for g, ig in zip(dalpha,implicit_grads):
			g.sub_(lr, ig)

		for v, g in zip(self.arch_parameters(), dalpha):
			v.grad = g.clone()
		if self.prox == 3:
			self._recover()
		return unrolled_loss

	def _compute_unrolled_model(self, p_train, q_train, r_train, labels_train, lr):
		inferences, regs = self(p_train, q_train, r_train)
		loss = self.compute_loss(inferences, labels_train, regs)
		theta = _concat(self.parameters())
		dtheta = _concat(torch.autograd.grad(loss, self.parameters())) + \
			self.reg * theta
		unrolled_model = self._construct_model_from_theta(
			theta.sub(lr, dtheta))
		return unrolled_model

	def _construct_model_from_theta(self, theta):
		model_new = self.new()
		model_dict = self.state_dict()
		params, offset = {}, 0
		for k,v in self.named_parameters():
			v_length = np.prod(v.size())
			params[k] = theta[offset: offset+v_length].view(v.size())
			offset += v_length

		assert offset == len(theta)
		model_dict.update(params)
		model_new.load_state_dict(model_dict)
		return model_new.cuda()

	def _hessian_vector_product(self, vector, p_train, q_train, r_train, labels_train, r=1e-2):
		R = r / _concat(vector).norm()
		for p,v in zip(self.parameters(), vector):
			p.data.add_(R, v)
		inferences, regs = self(p_train, q_train, r_train)
		loss = self.compute_loss(inferences, labels_train, regs)
		grads_p = torch.autograd.grad(loss, self.arch_parameters())

		for p,v in zip(self.parameters(), vector):
			p.data.sub_(2*R, v)
		inferences, regs = self(p_train, q_train, r_train)
		loss = self.compute_loss(inferences, labels_train, regs)
		grads_n = torch.autograd.grad(loss, self.arch_parameters())

		for p,v in zip(self.parameters(), vector):
			p.data.add_(R, v)

		return [(x-y).div_(2*R) for x,y in zip(grads_p,grads_n)]

class Network_MLP_Triple(Network_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg, prox):
		super(Network_MLP_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg, prox)

	def _initialize_alphas(self):
		self.mlp_p = nn.Sequential(
			nn.Linear(1, 8),
			nn.Tanh(),
			nn.Linear(8, 1)).cuda()
		self.mlp_q = nn.Sequential(
			nn.Linear(1, 8),
			nn.Tanh(),
			nn.Linear(8, 1)).cuda()
		self.mlp_r = nn.Sequential(
			nn.Linear(1, 8),
			nn.Tanh(),
			nn.Linear(8, 1)).cuda()

		self._arch_parameters = {}; self._arch_parameters['mlp'] = {}
		self._arch_parameters['mlp']['p'] = self.mlp_p
		self._arch_parameters['mlp']['q'] = self.mlp_q
		self._arch_parameters['mlp']['r'] = self.mlp_r
		self._arch_parameters['triple'] = initialize_alpha(len(PRIMITIVES_TRIPLE))

	def arch_parameters(self):
		return list(self._arch_parameters['mlp']['p'].parameters()) + \
			   list(self._arch_parameters['mlp']['q'].parameters()) + \
			   list(self._arch_parameters['mlp']['r'].parameters()) + \
			   [self._arch_parameters['triple']]

	def new(self):
		model_new = Network_MLP_Triple(self.num_ps, self.num_qs, self.num_rs, self.embedding_dim, self.reg, self.prox).cuda()
		for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
			x.data = y.data.clone()
			try:
				x.grad = y.grad.clone()
			except:
				pass
		return model_new

	def _binarize(self):
		self._cache = self._arch_parameters['triple'].clone()
		self._arch_parameters['triple'] = binarize(self._arch_parameters['triple'])

	def _recover(self):
		self._arch_parameters['triple'].data = self._cache.data.clone()

	def forward(self, ps, qs, rs):
		for i in range(len(PRIMITIVES_TRIPLE)):
			constrain(next(self._FC[i].parameters()))
		for i in range(len(self.arch_parameters())):
			if i == len(self.arch_parameters()) - 1: break
			if len(self.arch_parameters()[i].size()) == 1: continue
			constrain(self.arch_parameters()[i])

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		ps_embedding_trans = self._arch_parameters['mlp']['p'](ps_embedding.view(-1,1)).view(ps_embedding.size())
		qs_embedding_trans = self._arch_parameters['mlp']['q'](qs_embedding.view(-1,1)).view(qs_embedding.size())
		rs_embedding_trans = self._arch_parameters['mlp']['r'](rs_embedding.view(-1,1)).view(rs_embedding.size())

		inferences = MixedTriple(ps_embedding_trans, qs_embedding_trans, rs_embedding_trans,
			self._arch_parameters['triple'], self._FC)

		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs

	def genotype(self):
		genotype = PRIMITIVES_TRIPLE[self._arch_parameters['triple'].argmax().cpu().numpy()]
		genotype_p = F.softmax(self._arch_parameters['triple'], dim=-1)
		return genotype, genotype_p

class Network_MLP_Single_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, arch, reg, path):
		super(Network_MLP_Single_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self.arch = arch
		
		network_mlp = Network_MLP_Triple(num_ps, num_qs, num_rs, embedding_dim, reg)
		network_mlp.load_state_dict(torch.load(path))
		self.mlp_p = network_mlp._arch_parameters['mlp']['p']
		self.mlp_q = network_mlp._arch_parameters['mlp']['q']
		self.mlp_r = network_mlp._arch_parameters['mlp']['r']

		# self.mlp_p = nn.Sequential(
		# 	nn.Linear(1, 8),
		# 	nn.Tanh(),
		# 	nn.Linear(8, 1)).cuda()
		# self.mlp_q = nn.Sequential(
		# 	nn.Linear(1, 8),
		# 	nn.Tanh(),
		# 	nn.Linear(8, 1)).cuda()
		# self.mlp_r = nn.Sequential(
		# 	nn.Linear(1, 8),
		# 	nn.Tanh(),
		# 	nn.Linear(8, 1)).cuda()

		if arch == 'concat_concat':
			self._FC = nn.Linear(3*embedding_dim, 1, bias=False)
		elif 'concat' in arch:
			self._FC = nn.Linear(2*embedding_dim, 1, bias=False)
		else:
			self._FC = nn.Linear(embedding_dim, 1, bias=False)
		self._arch_parameters = list(self.mlp_p.parameters()) + list(self.mlp_q.parameters()) + \
			list(self.mlp_r.parameters())

	def train_parameters(self):
		return list(self._PsEmbedding.parameters()) + list(self._QsEmbedding.parameters()) + \
			list(self._RsEmbedding.parameters()) + list(self._FC.parameters())

	def forward(self, ps, qs, rs):
		constrain(next(self._FC.parameters()))

		for i in range(len(self._arch_parameters)):
			if len(self._arch_parameters[i].size()) == 1: continue
			constrain(self._arch_parameters[i])

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		ps_embedding_trans = self.mlp_p(ps_embedding.view(-1,1)).view(ps_embedding.size())
		qs_embedding_trans = self.mlp_q(qs_embedding.view(-1,1)).view(qs_embedding.size())
		rs_embedding_trans = self.mlp_r(rs_embedding.view(-1,1)).view(rs_embedding.size())

		inferences = self._FC(ops_triple(self.arch, ps_embedding_trans, 
			qs_embedding_trans, rs_embedding_trans))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + \
			torch.norm(rs_embedding))
		return inferences, regs

class AutoNeural_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(AutoNeural_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = nn.Sequential(
			nn.Linear(3*embedding_dim, 3*embedding_dim),
			nn.Sigmoid(),
			nn.Linear(3*embedding_dim, 1))

	def forward(self, ps, qs, rs):
		for p in self._FC.parameters():
			if len(p.size()) == 1: continue
			constrain(p)

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		inferences = self._FC(torch.cat([ps_embedding,qs_embedding,rs_embedding], dim=-1))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))

		return inferences, regs

	def embedding_parameters(self):
		return list(self._PsEmbedding.parameters()) + list(self._QsEmbedding.parameters()) + \
			list(self._RsEmbedding.parameters())

	def mlp_parameters(self):
		return self._FC.parameters()

class NAS_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, arch, reg):
		super(NAS_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = []

		for i in range(len(arch)):
			if i == 0:
				self._FC.append(nn.Linear(3*embedding_dim, int(arch[i])))
			else:
				self._FC.append(nn.Linear(int(arch[i-1]), int(arch[i])))
			self._FC.append(nn.ReLU())
		if len(self._FC) == 0:
			self._FC.append(nn.Linear(3*embedding_dim, 1, bias=False))
		else:
			self._FC.append(nn.Linear(arch[-1], 1, bias=False))
		self._FC = nn.Sequential(*self._FC)

	def forward(self, ps, qs, rs):
		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		inferences = self._FC(torch.cat([ps_embedding, qs_embedding, rs_embedding], dim=-1))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs
	










if __name__ == '__main__':

	# m = Network_MLP_Single(9433,1682,2,None,1e-5,'/data3/chenxiangning/models/binarydarts_mlp_ml-100k_2_Adagrad0.05_1e-05_1')
	# m = Conv(100,200,2,1e-5).cuda()

	m = Network_MLP(100,100,4,1e-5).cuda()
	for i in m.parameters():
		print(i.size())



