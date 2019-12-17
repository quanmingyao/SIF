import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models import PRIMITIVES_UNARY, PRIMITIVES_BINARY, PRIMITIVES_ASSIST, PRIMITIVES_TRIPLE, PRIMITIVES_NAS
from itertools import chain

# def sample_arch():
# 	arch = {}
# 	arch['unary'], arch['assist'] = {}, {}
# 	arch['unary']['p'] = PRIMITIVES_UNARY[np.random.randint(len(PRIMITIVES_UNARY))]
# 	arch['unary']['q'] = PRIMITIVES_UNARY[np.random.randint(len(PRIMITIVES_UNARY))]
# 	arch['assist']['p'] = PRIMITIVES_ASSIST[np.random.randint(len(PRIMITIVES_ASSIST))]
# 	arch['assist']['q'] = PRIMITIVES_ASSIST[np.random.randint(len(PRIMITIVES_ASSIST))]
# 	arch['binary'] = PRIMITIVES_BINARY[np.random.randint(len(PRIMITIVES_BINARY))]
# 	return arch

# def sample_arch_triple():
# 	arch = {}
# 	arch['unary'], arch['assist'] = {}, {}
# 	arch['unary']['p'] = PRIMITIVES_UNARY[np.random.randint(len(PRIMITIVES_UNARY))]
# 	arch['unary']['q'] = PRIMITIVES_UNARY[np.random.randint(len(PRIMITIVES_UNARY))]
# 	arch['unary']['r'] = PRIMITIVES_UNARY[np.random.randint(len(PRIMITIVES_UNARY))]
# 	arch['assist']['p'] = PRIMITIVES_ASSIST[np.random.randint(len(PRIMITIVES_ASSIST))]
# 	arch['assist']['q'] = PRIMITIVES_ASSIST[np.random.randint(len(PRIMITIVES_ASSIST))]
# 	arch['assist']['r'] = PRIMITIVES_ASSIST[np.random.randint(len(PRIMITIVES_ASSIST))]
# 	arch['triple'] = PRIMITIVES_TRIPLE[np.random.randint(len(PRIMITIVES_TRIPLE))]
# 	return arch

def sample_arch():
	arch = {}
	arch['mlp'] = {}
	arch['mlp']['p'] = nn.Sequential(
		nn.Linear(1, 8),
		nn.Tanh(),
		nn.Linear(8, 1)).cuda()
	arch['mlp']['q'] = nn.Sequential(
		nn.Linear(1, 8),
		nn.Tanh(),
		nn.Linear(8, 1)).cuda()
	arch['binary'] = PRIMITIVES_BINARY[np.random.randint(len(PRIMITIVES_BINARY))]
	return arch

def sample_arch_triple():
	arch = {}
	arch['mlp'] = {}
	arch['mlp']['p'] = nn.Sequential(
		nn.Linear(1, 8),
		nn.Tanh(),
		nn.Linear(8, 1)).cuda()
	arch['mlp']['q'] = nn.Sequential(
		nn.Linear(1, 8),
		nn.Tanh(),
		nn.Linear(8, 1)).cuda()
	arch['mlp']['r'] = nn.Sequential(
		nn.Linear(1, 8),
		nn.Tanh(),
		nn.Linear(8, 1)).cuda()
	arch['triple'] = PRIMITIVES_TRIPLE[np.random.randint(len(PRIMITIVES_TRIPLE))]
	return arch

def sample_arch_nas():
	arch = [np.random.choice(PRIMITIVES_NAS) for _ in range(5)]
	return rectify_arch(arch)

def rectify_arch(arch):
	pos = np.where(np.array(arch) == 0)[0]
	if not pos.size == 0: arch = arch[:pos[0]]
	return arch

def update_arch(arch, cfg):
	flag = 0
	for p in arch.parameters():
		num = p.view(-1).size(0)
		p.data.add_(-p).add_(torch.tensor(cfg[flag:flag+num]).float().cuda().view(p.size()))
		flag += num

class Controller(nn.Module):

	def __init__(self, choice, triple=False):
		super(Controller, self).__init__()
		self.triple = triple
		if triple:
			self.space = max(len(PRIMITIVES_UNARY), len(PRIMITIVES_ASSIST), \
				len(PRIMITIVES_TRIPLE))
			self.num_action = 7
		else:
			self.space = max(len(PRIMITIVES_UNARY), len(PRIMITIVES_ASSIST), \
				len(PRIMITIVES_BINARY))
			self.num_action = 5
		self.choice = choice
		if choice == 'RNN':
			self.controller = nn.RNNCell(input_size=self.space, hidden_size=self.space)
		elif choice == 'PURE':
			self._arch_parameters = []
			for _ in range(self.num_action):
				alpha = torch.ones([1,self.space], dtype=torch.float, device='cuda') / self.space
				# alpha = alpha + torch.randn(self.space, device='cuda') * 1e-2
				self._arch_parameters.append(Variable(alpha, requires_grad=True))

	def arch_parameters(self):
		return self._arch_parameters

	def forward(self):
		if self.choice == 'RNN':
			input0 = torch.ones([1,self.space]) / self.space / 10.0
			input0 = input0.cuda()
			h = torch.zeros([1, self.space]).cuda()
			inferences = []
			for i in range(self.num_action):
				if i == 0:
					h = self.controller(input0, h)
				else:
					h = self.controller(h, h)
				inferences.append(h)
			return inferences
		elif self.choice == 'PURE':
			return self._arch_parameters

	def compute_loss(self, rewards):
		inferences = torch.cat(self(), dim=0).repeat(len(self.archs), 1)
		self.archs = torch.tensor(list(chain(*self.archs))).cuda()
		rewards = torch.reshape(torch.tensor(rewards), [-1,1])
		rewards = torch.reshape(rewards.repeat(1,self.num_action), [-1,1]).cuda()
		return torch.mean(rewards * F.cross_entropy(inferences, self.archs))

	def print_prob(self):
		inferences = self()
		for infer in inferences:
			infer = F.softmax(infer, dim=-1).cpu().detach().numpy()

	def sample_arch(self, batch_size):
		self.archs, archs = [], []
		inferences = self()
		batch_count = 0
		while batch_count < batch_size:
			tmp = []
			arch = {}
			arch['unary'], arch['assist'] = {}, {}
			for action_count, infer in enumerate(inferences):
				infer = torch.squeeze(infer)
				if self.triple:
					if action_count == 0:
						p = F.softmax(infer[:len(PRIMITIVES_UNARY)], dim=-1).cpu().detach().numpy()
						choice = np.random.choice(len(PRIMITIVES_UNARY), p=p)
						arch['unary']['p'] = PRIMITIVES_UNARY[choice]
					elif action_count == 1:
						p = F.softmax(infer[:len(PRIMITIVES_UNARY)], dim=-1).cpu().detach().numpy()
						choice = np.random.choice(len(PRIMITIVES_UNARY), p=p)
						arch['unary']['q'] = PRIMITIVES_UNARY[choice]
					elif action_count == 2:
						p = F.softmax(infer[:len(PRIMITIVES_UNARY)], dim=-1).cpu().detach().numpy()
						choice = np.random.choice(len(PRIMITIVES_UNARY), p=p)
						arch['unary']['r'] = PRIMITIVES_UNARY[choice]
					elif action_count == 3:
						p = F.softmax(infer[:len(PRIMITIVES_ASSIST)], dim=-1).cpu().detach().numpy()
						choice = np.random.choice(len(PRIMITIVES_ASSIST), p=p)
						arch['assist']['p'] = PRIMITIVES_ASSIST[choice]
					elif action_count == 4:
						p = F.softmax(infer[:len(PRIMITIVES_ASSIST)], dim=-1).cpu().detach().numpy()
						choice = np.random.choice(len(PRIMITIVES_ASSIST), p=p)
						arch['assist']['q'] = PRIMITIVES_ASSIST[choice]
					elif action_count == 5:
						p = F.softmax(infer[:len(PRIMITIVES_ASSIST)], dim=-1).cpu().detach().numpy()
						choice = np.random.choice(len(PRIMITIVES_ASSIST), p=p)
						arch['assist']['r'] = PRIMITIVES_ASSIST[choice]
					elif action_count == 6:
						p = F.softmax(infer[:len(PRIMITIVES_TRIPLE)], dim=-1).cpu().detach().numpy()
						choice = np.random.choice(len(PRIMITIVES_TRIPLE), p=p)
						arch['triple'] = PRIMITIVES_TRIPLE[choice]

				else:
					if action_count == 0:
						p = F.softmax(infer[:len(PRIMITIVES_UNARY)], dim=-1).cpu().detach().numpy()
						choice = np.random.choice(len(PRIMITIVES_UNARY), p=p)
						arch['unary']['p'] = PRIMITIVES_UNARY[choice]
					elif action_count == 1:
						p = F.softmax(infer[:len(PRIMITIVES_UNARY)], dim=-1).cpu().detach().numpy()
						choice = np.random.choice(len(PRIMITIVES_UNARY), p=p)
						arch['unary']['q'] = PRIMITIVES_UNARY[choice]
					elif action_count == 2:
						p = F.softmax(infer[:len(PRIMITIVES_ASSIST)], dim=-1).cpu().detach().numpy()
						choice = np.random.choice(len(PRIMITIVES_ASSIST), p=p)
						arch['assist']['p'] = PRIMITIVES_ASSIST[choice]
					elif action_count == 3:
						p = F.softmax(infer[:len(PRIMITIVES_ASSIST)], dim=-1).cpu().detach().numpy()
						choice = np.random.choice(len(PRIMITIVES_ASSIST), p=p)
						arch['assist']['q'] = PRIMITIVES_ASSIST[choice]
					elif action_count == 4:
						p = F.softmax(infer[:len(PRIMITIVES_BINARY)], dim=-1).cpu().detach().numpy()
						choice = np.random.choice(len(PRIMITIVES_BINARY), p=p)
						arch['binary'] = PRIMITIVES_BINARY[choice]
				tmp.append(choice)
			if not arch in archs:
				archs.append(arch)
				self.archs.append(tmp)
				batch_count += 1
		return archs

class Controller_NAS(Controller):

	def __init__(self, choice, triple=False):
		super(Controller_NAS, self).__init__(choice, triple)
		self.space = len(PRIMITIVES_NAS)
		self.num_action = 5
		self.choice = choice
		if choice == 'RNN':
			self.controller = nn.RNNCell(input_size=self.space, hidden_size=self.space)
		elif choice == 'PURE':
			self._arch_parameters = []
			for _ in range(self.num_action):
				alpha = torch.ones([1,self.space], dtype=torch.float, device='cuda') / self.space
				alpha = alpha + torch.randn(self.space, device='cuda') * 1e-2
				self._arch_parameters.append(Variable(alpha, requires_grad=True))

	def sample_arch(self, batch_size):
		self.archs, archs = [], []
		inferences = self()
		batch_count = 0
		while batch_count < batch_size:
			tmp = []
			arch = []
			for action_count, infer in enumerate(inferences):
				infer = torch.squeeze(infer)
				choice = np.random.choice(len(PRIMITIVES_NAS), p=F.softmax(infer, dim=-1).cpu().detach().numpy())
				arch.append(PRIMITIVES_NAS[choice])
				tmp.append(choice)
			if not arch in archs:
				archs.append(rectify_arch(arch))
				self.archs.append(tmp)
				batch_count += 1
		return archs
	





if __name__ == '__main__':
	# for _ in range(10):
	# 	print(sample_arch())
	a = nn.Sequential(
			nn.Linear(1, 8),
			nn.Tanh(),
			nn.Linear(8, 1)).cuda()
	c = np.random.randn(25)

	for p in a.parameters(): print(p)
	update_arch(a, c)
	for p in a.parameters(): print(p)





