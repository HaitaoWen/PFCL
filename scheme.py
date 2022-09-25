import math
import geoopt
import quadprog
import numpy as np
import torch.optim
from utils import *
from tqdm import tqdm
from dcd import DCD
from copy import deepcopy
from rsgd import RiemannianSGD
from collections import OrderedDict
from torch.utils.data import DataLoader
from sampler import MemoryBatchSampler


def online_learning_modify_window(**kwargs):
    scheme = kwargs['scheme']
    taskid = scheme.taskid
    if taskid == 1:
        scheme.optimizer = torch.optim.SGD(scheme.model.parameters(), lr=args.lr, momentum=args.momentum,
                                           weight_decay=0)

    if taskid > 1:
        if args.online:
            args.epochs = 1


class FineTune:
    """
    base Class and lower bound for continual learning.
    """

    def __init__(self, model, traindata, taskid):
        self.loss_ = 0
        self.iterations = 0
        self.taskid = taskid
        self.progress = None
        self.traindata = traindata
        if nni_params:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cuda', args.gpuid)
        self.model = model.to(self.device)
        self.minclass, self.maxclass = get_minmax_class(self.taskid)
        self.trainloader = DataLoader(traindata, batch_size=args.bs, shuffle=True, num_workers=4)
        self.criterion = self.define_loss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                         weight_decay=args.decay)
        online_learning_modify_window(**{'scheme': self})
        if args.steps is not None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.steps,
                                                                  gamma=args.gamma)
        else:
            self.scheduler = None
        self.model.train()

    def train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt or 'nni' in args.opt)
        self.progress.set_description('train')
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                y_ = self.model(x)
                loss = self.criterion(y_, y, t)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item()})
                self.progress.update(1)
            if self.scheduler is not None:
                self.scheduler.step()
        self.progress.close()
        return self.model

    # single head loss
    def single_head_loss(self, predict, target, taskids):
        """
        1) activate corresponding head
        """
        y_, y = activate_head(self.minclass, self.maxclass, predict, target)
        loss = torch.nn.functional.cross_entropy(y_, y)
        return loss

    @staticmethod
    def multi_head_loss(predict, target, taskids):
        """
        1) unique task indices
        2) iterate each head and activate it
        3) compute specific head loss
        4) average multi heads' loss
        """
        loss = 0
        taskids_ = torch.unique(taskids)
        for t in taskids_:
            mask = (t == taskids)
            minclass, maxclass = get_minmax_class(t)
            y_, y = activate_head(minclass, maxclass, predict[mask], target[mask])
            loss = loss + torch.nn.functional.cross_entropy(y_, y, reduction='sum')
        loss = loss / predict.shape[0]
        return loss

    def define_loss(self):
        if args.scenario == 'task':
            if args.memory > 0:
                loss_function = self.multi_head_loss
            else:
                loss_function = self.single_head_loss
        elif args.scenario == 'domain':
            loss_function = self.single_head_loss
        elif args.scenario == 'class':
            loss_function = self.single_head_loss
        else:
            raise ValueError("Unsupported {} scenario".format(args.scenario))
        return loss_function

    @property
    def loss(self):
        return self.loss_

    @loss.setter
    def loss(self, tmp):
        loss = self.loss_ * self.iterations + tmp
        self.iterations += 1
        self.loss_ = loss / self.iterations

    def detach_state(self):
        state = OrderedDict()
        for key, value in self.model.state_dict().items():
            state[key] = value.clone().detach()
        return state

    def clone_parameters(self):
        parameters = OrderedDict()
        for key, value in self.model.named_parameters():
            parameters[key] = value.clone()
        return parameters

    def detach_parameters(self):
        parameters = OrderedDict()
        for key, value in self.model.named_parameters():
            parameters[key] = value.clone().detach()
        return parameters

    @staticmethod
    def state_detach(state):
        state_ = OrderedDict()
        for key, value in state.items():
            state_[key] = value.clone().detach()
        return state_


class MultiTask(FineTune):
    """
    upper bound for continual learning.
    """

    def __init__(self, model, traindata, taskid):
        super(MultiTask, self).__init__(model, traindata, taskid)
        if ('CIFAR' in args.dataset) or ('ImageNet' in args.dataset):
            dataset = traindata[0]
            dataset.concat(traindata[1:])
        elif 'MNIST' in args.dataset:
            dataset = []
            x, y, _ = traindata.dataset
            for task_index in range(args.tasks):
                trans = traindata.get_task_transformation(task_index)
                if not ('slient' in args.opt or 'nni' in args.opt):
                    print('{} {}'.format(args.dataset, task_index + 1))
                for x_, y_ in zip(x, y):
                    x_ = Image.fromarray(x_.astype("uint8"))
                    x_t = trans(x_).squeeze()
                    dataset.append([x_t, y_, task_index])
            dataset = MNISTData(dataset)
        else:
            raise ValueError('Unsupported dataset: {}'.format(args.dataset))
        self.criterion = self.define_loss()
        self.trainloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                         weight_decay=args.decay)

    def define_loss(self):
        if args.scenario == 'task':
            loss_function = self.multi_head_loss
        elif args.scenario == 'domain':
            loss_function = self.multi_task_loss
        elif args.scenario == 'class':
            loss_function = self.multi_task_loss
        else:
            raise ValueError("Unsupported {} scenario".format(args.scenario))
        return loss_function

    @staticmethod
    def multi_task_loss(predict, target, taskids):
        loss = torch.nn.functional.cross_entropy(predict, target)
        return loss


class EWC(FineTune):
    r"""
    [1]Kirkpatrick J, Pascanu R, Rabinowitz N, et al.
    Overcoming catastrophic forgetting in neural networks[J].
    Proceedings of the national academy of sciences, 2017, 114(13): 3521-3526.

    [2]Schwarz J, Czarnecki W, Luketina J, et al.
    Progress & compress: A scalable framework for continual learning[C]
    International Conference on Machine Learning. PMLR, 2018: 4528-4537.

    Args:
        lambd: the coefficient of EWC loss
        online: online EWC or not
        gamma: EMA coefficient of online EWC, memory strength of old tasks
    """

    def __init__(self, model, traindata, taskid):
        super(EWC, self).__init__(model, traindata, taskid)
        self.epsilon = 1e-32
        self.lambd = args.lambd

        self.online = args.online
        self.gamma = args.gamma

    def train(self):
        self.ewc_train()
        self.compute_fisher()
        return self.model

    def ewc_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt or 'nni' in args.opt)
        self.progress.set_description('train')
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                y_ = self.model(x)
                loss = self.criterion(y_, y, t)
                if self.taskid > 1:
                    ewc_loss = self.ewc_loss() * self.lambd
                    loss += ewc_loss
                else:
                    ewc_loss = 0
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item(),
                                           'ewcloss': ewc_loss.item() if self.taskid > 1 else 0})
                self.progress.update(1)
        self.progress.close()

    def compute_fisher(self):
        # compute empirical fisher matrix
        fisher = {}
        self.model.eval()
        # theoretically it should compute each sample's gradient,
        # here we compute each batch's gradient for saving time.
        progress = tqdm(self.trainloader, disable='slient' in args.opt or 'nni' in args.opt)
        progress.set_description('fisher')
        for x, y, t in progress:
            t = t + 1
            x = x.to(self.device)
            y = y.to(self.device)
            y_ = self.model(x)
            loss = self.criterion(y_, y, t)
            self.model.zero_grad()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and (p.grad is not None):
                    if n not in fisher:
                        fisher[n] = p.grad.clone().detach() ** 2
                    else:
                        fisher[n] += p.grad.clone().detach() ** 2
        min_value = []
        max_value = []
        # expection
        for n, p in fisher.items():
            scaled = p / len(self.trainloader)
            fisher[n] = scaled
            min_value.append(torch.min(scaled).unsqueeze(dim=0))
            max_value.append(torch.max(scaled).unsqueeze(dim=0))
        min_value = torch.min(torch.cat(min_value))
        max_value = torch.max(torch.cat(max_value))
        # normalize
        for n, p in fisher.items():
            fisher[n] = (p - min_value) / (max_value - min_value + self.epsilon)
        param_ = {n: p.clone().detach() for n, p in self.model.named_parameters()}
        if hasattr(self.model, 'fisher_param'):
            if self.online:
                for n, p in self.model.fisher_param['fisher'].items():
                    self.model.fisher_param['fisher'][n] += fisher[n] * self.gamma
                self.model.fisher_param['param_'] = param_
            else:
                self.model.fisher_param[self.taskid] = {'fisher': fisher, 'param_': param_}
        else:
            if self.online:
                self.model.__setattr__('fisher_param', {'fisher': fisher, 'param_': param_})
            else:
                self.model.__setattr__('fisher_param', {self.taskid: {'fisher': fisher, 'param_': param_}})
        self.model.train()

    def ewc_loss(self):
        loss = []
        if self.online:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    f = self.model.fisher_param['fisher'][n] * self.gamma
                    p_ = self.model.fisher_param['param_'][n]
                    loss.append((f * ((p - p_) ** 2)).sum())
            return sum(loss)
        else:
            for task in range(1, self.taskid):
                fisher = self.model.fisher_param[task]['fisher']
                param_ = self.model.fisher_param[task]['param_']
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        f = fisher[n]
                        p_ = param_[n]
                        loss.append((f * ((p - p_) ** 2)).sum())
            return sum(loss) / (self.taskid - 1)


class ER(FineTune):
    """
    Chaudhry A, Rohrbach M, Elhoseiny M, et al.
    On tiny episodic memories in continual learning[J].
    arXiv preprint arXiv:1902.10486, 2019.

    Args:
        memory: the number of samples sampled for a class
        opt: list type, default sampling mode is MoF(offline);
             if 'random_sample' is in this list, sampling mode is random(offline);
    """

    def __init__(self, model, traindata, taskid):
        super(ER, self).__init__(model, traindata, taskid)
        assert args.memory > 0, 'ER based scheme needs memory to store samples of old tasks!'
        if taskid == 1:
            return
        if args.mcat:
            cur_data_num = len(self.traindata)
            sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=[i for i in range(cur_data_num)])
            batch_sampler = MemoryBatchSampler(sampler, batch_size=args.bs, mem_size=len(memory.x),
                                               mem_batch_size=args.mbs, drop_last=False)
        if args.dataset == 'CIFAR100':
            dataset = CIFARData(self.traindata, memory)
        elif 'MNIST' in args.dataset:
            dataset = []
            trans = self.traindata.trsf
            _x, _y, _t = self.traindata._x, self.traindata._y, self.traindata._t
            for x, y, t in zip(_x, _y, _t):
                x = Image.fromarray(x.astype("uint8"))
                x_t = trans(x).squeeze()
                dataset.append([x_t, y, int(t)])
            dataset.extend(memory.x)
            dataset = MNISTData(dataset)
        elif 'ImageNet' in args.dataset:
            x = traindata._x.tolist() + memory.x
            y = traindata._y.tolist() + memory.y
            t = traindata._t.tolist() + memory.t
            dataset = ImageNetData(x, y, t, traindata.trsf)
        else:
            raise ValueError('Unsupported dataset: {}'.format(args.dataset))
        if args.mcat:
            self.trainloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)
        else:
            self.trainloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=4)

    def train(self):
        super(ER, self).train()
        self.construct_exemplar_set()
        return self.model

    @staticmethod
    def get_representations_hook(model):
        def hook(module, input, output):
            model.representations = input

        return hook

    def register_fc_hook(self, call_back):
        setattr(self.model, 'representations', None)
        hook_handle = None
        for name, module in self.model.named_modules():
            if name == 'fc':
                hook_handle = module.register_forward_hook(call_back(self.model))
        return hook_handle

    def mean_based_sample(self, _x, _y, trsf):
        x, y, t = [], [], []
        classes = np.unique(_y)
        for c in classes:
            mask = c == _y
            images = _x[mask]
            imagesdata = Images(images, trsf)
            loader = DataLoader(imagesdata, args.bs, shuffle=False)
            sample_indices = []
            representations = []
            for x_, index in loader:
                x_ = x_.to(self.device)
                sample_indices.append(index)
                with torch.no_grad():
                    self.model(x_)
                representations.append(self.model.representations[0])
            sample_indices = torch.cat(sample_indices, dim=0)
            representations = torch.cat(representations, dim=0)
            representations = torch.nn.functional.normalize(representations, p=2, dim=1)
            mean_representation = representations.mean(dim=0)
            mean_representation = torch.nn.functional.normalize(mean_representation, p=2, dim=0)
            sampled_indices_ = []
            while True:
                if len(sampled_indices_) == 0:
                    representations_ = representations.clone()
                    tmp_indices = torch.arange(representations.size(0))
                else:
                    mask = torch.zeros(representations.size(0)).bool()
                    mask[sampled_indices_] = True
                    # left samples
                    representations_ = representations[~mask, :].clone()
                    tmp_indices = torch.arange(representations.size(0))
                    tmp_indices = tmp_indices[~mask]
                    # sampled samples
                    representations__ = representations[mask, :].clone()
                    representations__ = torch.sum(representations__, dim=0)
                    representations_ += representations__
                    representations_ /= (len(sampled_indices_) + 1)
                diff_representation = representations_ - mean_representation
                diff_representation_norm = torch.norm(diff_representation, p=2, dim=1)
                try:
                    index = torch.argmin(diff_representation_norm, dim=0)
                except RuntimeError:
                    break  # if the number of images in dataloader < M_total
                sampled_indices_.append(tmp_indices[index].item())
                if len(sampled_indices_) >= args.memory:
                    break
            indices = sample_indices[sampled_indices_]
            for i in indices:
                x.append(images[i]), y.append(c), t.append(self.taskid - 1)
        return x, y, t

    def random_based_sample(self, _x, _y):
        x, y, t = [], [], []
        classes = np.unique(_y)
        for c in classes:
            mask = c == _y
            indices = np.where(mask)[0]
            indices = np.random.choice(indices, size=args.memory, replace=False)
            for index in indices:
                x.append(_x[index])
                y.append(_y[index])
                t.append(self.taskid - 1)
        return x, y, t

    def construct_exemplar_set(self):
        self.model.eval()
        # register hook to get representations
        hook_handle = self.register_fc_hook(self.get_representations_hook)
        trsf = self.traindata.trsf
        _x, _y = self.traindata._x, self.traindata._y
        if 'herding' in args.opt:
            x, y, t = self.mean_based_sample(_x, _y, trsf)
        else:
            x, y, t = self.random_based_sample(_x, _y)
        hook_handle.remove()
        if args.dataset == 'CIFAR100':
            x = np.stack(x, axis=0)
            y = np.stack(y, axis=0)
            t = np.stack(t, axis=0)
            memory.x = np.concatenate((memory.x, x), axis=0) if memory.x is not None else x
            memory.y = np.concatenate((memory.y, y), axis=0) if memory.y is not None else y
            memory.t = np.concatenate((memory.t, t), axis=0) if memory.t is not None else t
        elif 'MNIST' in args.dataset:
            x__ = []
            for x_, y_, t_ in zip(x, y, t):
                x_ = Image.fromarray(x_.astype("uint8"))
                x_ = trsf(x_).squeeze()
                x__.append([x_, y_, t_])
            if memory.x is None:
                memory.x = x__
            else:
                memory.x.extend(x__)
        elif 'ImageNet' in args.dataset:
            if memory.x is None:
                memory.x = x
                memory.y = y
                memory.t = t
            else:
                memory.x += x
                memory.y += y
                memory.t += t
        else:
            raise ValueError('Unsupported dataset: {}'.format(args.dataset))


class StableSGD(FineTune):
    r"""
    Mirzadeh S I, Farajtabar M, Pascanu R, et al.
    Understanding the role of training regimes in continual learning[J].
    arXiv preprint arXiv:2006.06958, 2020.

    Args:
        gamma: coefficient of Exponential Mean Average
    """

    def __init__(self, model, traindata, taskid):
        super(StableSGD, self).__init__(model, traindata, taskid)
        lr = max(args.lr * args.gamma ** (taskid - 1), 0.00005)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=args.decay,
                                         momentum=args.momentum)


class GEM(ER):
    r"""
    [1] Lopez-Paz D, Ranzato M A.
    Gradient episodic memory for continual learning[J].
    Advances in neural information processing systems, 2017, 30: 6467-6476.

    [2] Chaudhry A, Ranzato M A, Rohrbach M, et al.
    Efficient lifelong learning with a-gem[J].
    arXiv preprint arXiv:1812.00420, 2018.

    code is mostly borrowed from:
    https://github.com/facebookresearch/GradientEpisodicMemory

    Args:
        sigma: memory strength, work with GEM
        opt: default scheme is GEM,
             if 'A-GEM' is in opt, the average gradient of old tasks will
             be as reference gradient, this will save time and space.
    """

    def __init__(self, model, traindata, taskid):
        super(GEM, self).__init__(model, traindata, taskid)
        if self.taskid == 1:
            return
        self.eps = 1e-3
        self.sigma = args.sigma
        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        if 'A-GEM' in args.opt:
            self.grads = torch.zeros(sum(self.grad_dims), 2).to(self.device)
        else:
            self.grads = torch.zeros(sum(self.grad_dims), self.taskid).to(self.device)

    def train(self):
        if self.taskid == 1:
            FineTune.train(self)
        else:
            self.gem_train()
        self.construct_exemplar_set()
        return self.model

    def gem_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt or 'nni' in args.opt)
        self.progress.set_description('train')
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                mask = t == self.taskid
                # current task gradient
                y_ = self.model(x)
                loss = self.criterion(y_[mask], y[mask], t[mask])
                self.model.zero_grad()
                loss.backward(retain_graph=True)
                self.store_grad(0)
                # old tasks' gradients
                if 'A-GEM' in args.opt:
                    loss = self.criterion(y_[~mask], y[~mask], t[~mask])
                    self.model.zero_grad()
                    loss.backward()
                    self.store_grad(1)
                else:
                    old_taskid = t[~mask].unique()
                    for taskid in old_taskid:
                        mask_ = taskid == t[~mask]
                        loss = self.criterion(y_[~mask][mask_], y[~mask][mask_], t[~mask][mask_])
                        self.model.zero_grad()
                        if taskid != old_taskid[-1]:
                            loss.backward(retain_graph=True)
                        else:
                            loss.backward()
                        self.store_grad(taskid)
                dotp = torch.mm(self.grads[:, 0].unsqueeze(0), self.grads[:, 1:])
                if (dotp < 0).sum() != 0:
                    self.project2cone2()
                    self.overwrite_grad()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item()})
                self.progress.update(1)
        self.progress.close()

    def store_grad(self, index):
        self.grads[:, index].fill_(0.0)
        cnt = 0
        for param in self.model.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
                end = sum(self.grad_dims[:cnt + 1])
                self.grads[beg: end, index].copy_(param.grad.data.view(-1))
            cnt += 1

    def project2cone2(self):
        if 'A-GEM' in args.opt:
            gradient = self.grads[:, 0].unsqueeze(1)
            memories = self.grads[:, 1].unsqueeze(1)
            pdot = torch.mm(gradient.T, memories)
            norm = torch.mm(memories.T, memories)
            new_gradient = gradient - pdot * memories / norm
            self.grads[:, 0] = new_gradient.squeeze(1)
        else:
            memories = self.grads[:, 1:]
            gradient = self.grads[:, 0].unsqueeze(1)
            memories_np = memories.cpu().t().double().numpy()
            gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
            t = memories_np.shape[0]
            P = np.dot(memories_np, memories_np.transpose())
            P = 0.5 * (P + P.transpose()) + np.eye(t) * self.eps
            q = np.dot(memories_np, gradient_np) * -1
            G = np.eye(t)
            h = np.zeros(t) + self.sigma
            v = quadprog.solve_qp(P, q, G, h)[0]
            x = np.dot(v, memories_np) + gradient_np
            self.grads[:, 0] = torch.tensor(x).view(-1)

    def overwrite_grad(self):
        cnt = 0
        for param in self.model.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
                en = sum(self.grad_dims[:cnt + 1])
                this_grad = self.grads[beg: en, 0].contiguous().view(param.grad.data.size())
                param.grad.data.copy_(this_grad)
            cnt += 1


class AFEC(FineTune):
    """
    Wang L, Zhang M, Jia Z, et al.
    Afec: Active forgetting of negative transfer in continual learning[J].
    Advances in Neural Information Processing Systems, 2021, 34: 22379-22391.
    """
    def __init__(self, model, traindata, taskid):
        super(AFEC, self).__init__(model, traindata, taskid)
        assert args.lambd >= 0, 'coefficient of ewc loss'
        assert args.beta >= 0, 'coefficient of emp loss'
        self.epsilon = 1e-32
        self.beta = args.beta
        self.lambd = args.lambd

    def train(self):
        if self.taskid == 1:
            super(AFEC, self).train()
            emp_model = deepcopy(self.model)
            setattr(memory, 'emp_model', emp_model)
        else:
            # finetune emp_model on the new task
            self.emp_model_train()
            # compute fisher matrix of emp_model
            self.compute_emp_fisher()
            self.afec_train()
        self.compute_fisher()
        return self.model

    def afec_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt or 'nni' in args.opt)
        self.progress.set_description('afec train')
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                y_ = self.model(x)
                loss = self.criterion(y_, y, t)
                emp_loss = self.emp_loss() * self.beta
                ewc_loss = self.ewc_loss() * self.lambd
                loss = loss + ewc_loss + emp_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item(),
                                           'ewcloss': ewc_loss.item(),
                                           'emploss': emp_loss.item()})
                self.progress.update(1)
            if self.scheduler is not None:
                self.scheduler.step()
        self.progress.close()

    def emp_model_train(self):
        progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                        disable='slient' in args.opt or 'nni' in args.opt)
        progress.set_description('emp train')
        optimizer = torch.optim.SGD(memory.emp_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.decay)
        if args.steps is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.gamma)
        else:
            scheduler = None
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                y_ = memory.emp_model(x)
                loss = self.criterion(y_, y, t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.loss = loss.item()
                progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                      'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                                      'avgloss': round(self.loss, 3), 'loss': loss.item()})
                progress.update(1)
            if scheduler is not None:
                scheduler.step()
        progress.close()

    def compute_emp_fisher(self):
        # compute empirical fisher matrix
        fisher = {}
        memory.emp_model.eval()
        # theoretically it should compute each sample's gradient,
        # here we compute each batch's gradient for saving time.
        progress = tqdm(self.trainloader, disable='slient' in args.opt or 'nni' in args.opt)
        progress.set_description('emp fisher')
        for x, y, t in progress:
            t = t + 1
            x = x.to(self.device)
            y = y.to(self.device)
            y_ = memory.emp_model(x)
            loss = self.criterion(y_, y, t)
            memory.emp_model.zero_grad()
            loss.backward()
            for n, p in memory.emp_model.named_parameters():
                if p.requires_grad and (p.grad is not None):
                    if n not in fisher:
                        fisher[n] = p.grad.clone().detach() ** 2
                    else:
                        fisher[n] += p.grad.clone().detach() ** 2
        min_value = []
        max_value = []
        # expection
        for n, p in fisher.items():
            scaled = p / len(self.trainloader)
            fisher[n] = scaled
            min_value.append(torch.min(scaled).unsqueeze(dim=0))
            max_value.append(torch.max(scaled).unsqueeze(dim=0))
        min_value = torch.min(torch.cat(min_value))
        max_value = torch.max(torch.cat(max_value))
        # normalize
        for n, p in fisher.items():
            fisher[n] = (p - min_value) / (max_value - min_value + self.epsilon)
        param_ = {n: p.clone().detach() for n, p in memory.emp_model.named_parameters()}
        if hasattr(memory.emp_model, 'fisher_param'):
            for n, p in memory.emp_model.fisher_param['fisher'].items():
                memory.emp_model.fisher_param['fisher'][n] = fisher[n]
            memory.emp_model.fisher_param['param_'] = param_
        else:
            memory.emp_model.__setattr__('fisher_param', {'fisher': fisher, 'param_': param_})
        memory.emp_model.train()

    def compute_fisher(self):
        # compute empirical fisher matrix
        fisher = {}
        self.model.eval()
        # theoretically it should compute each sample's gradient,
        # here we compute each batch's gradient for saving time.
        progress = tqdm(self.trainloader, disable='slient' in args.opt or 'nni' in args.opt)
        progress.set_description('ewc fisher')
        for x, y, t in progress:
            t = t + 1
            x = x.to(self.device)
            y = y.to(self.device)
            y_ = self.model(x)
            loss = self.criterion(y_, y, t)
            self.model.zero_grad()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and (p.grad is not None):
                    if n not in fisher:
                        fisher[n] = p.grad.clone().detach() ** 2
                    else:
                        fisher[n] += p.grad.clone().detach() ** 2
        min_value = []
        max_value = []
        # expection
        for n, p in fisher.items():
            scaled = p / len(self.trainloader)
            fisher[n] = scaled
            min_value.append(torch.min(scaled).unsqueeze(dim=0))
            max_value.append(torch.max(scaled).unsqueeze(dim=0))
        min_value = torch.min(torch.cat(min_value))
        max_value = torch.max(torch.cat(max_value))
        # normalize
        for n, p in fisher.items():
            fisher[n] = (p - min_value) / (max_value - min_value + self.epsilon)
        param_ = {n: p.clone().detach() for n, p in self.model.named_parameters()}
        if hasattr(self.model, 'fisher_param'):
            for n, p in self.model.fisher_param['fisher'].items():
                self.model.fisher_param['fisher'][n] = (self.model.fisher_param['fisher'][n] * (self.taskid - 1) +
                                                        fisher[n]) / self.taskid
            self.model.fisher_param['param_'] = param_
        else:
            self.model.__setattr__('fisher_param', {'fisher': fisher, 'param_': param_})
        self.model.train()

    def emp_loss(self):
        loss = []
        for n, p in self.model.named_parameters():
            if 'fc' in n:
                continue
            if p.requires_grad:
                f = memory.emp_model.fisher_param['fisher'][n]
                p_ = memory.emp_model.fisher_param['param_'][n]
                loss.append((f * ((p - p_) ** 2)).sum())
        return sum(loss)

    def ewc_loss(self):
        loss = []
        for n, p in self.model.named_parameters():
            if 'fc' in n:
                continue
            if p.requires_grad:
                f = self.model.fisher_param['fisher'][n]
                p_ = self.model.fisher_param['param_'][n]
                loss.append((f * ((p - p_) ** 2)).sum())
        return sum(loss)


class OGD(FineTune):
    """
    Farajtabar M, Azizan N, Mott A, et al.
    Orthogonal gradient descent for continual learning[C]//
    International Conference on Artificial Intelligence and Statistics.
    PMLR, 2020: 3762-3773.
    """
    def __init__(self, model, traindata, taskid):
        super(OGD, self).__init__(model, traindata, taskid)
        assert args.memory > 0, 'number of samples for computing gradients'
        self.bias = None

    def train(self):
        if self.taskid == 1:
            super(OGD, self).train()
        else:
            self.update_bias()
            self.ogd_train()
        self.sample()
        self.store_grad()
        return self.model

    def ogd_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt or 'nni' in args.opt)
        self.progress.set_description('train')
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                y_ = self.model(x)
                loss = self.criterion(y_, y, t)
                self.optimizer.zero_grad()
                loss.backward()
                self.project()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item()})
                self.progress.update(1)
            if self.scheduler is not None:
                self.scheduler.step()
        self.progress.close()
        return self.model

    def store_grad(self):
        self.model.eval()
        if not hasattr(memory, 'grad'):
            memory.__setattr__('grad', [])
        if 'ImageNet' in args.dataset:
            dataset = ImageNetData(memory.x, memory.y, memory.t, self.traindata.trsf)
        else:
            dataset = Memorydata(memory, self.traindata.trsf)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        for x, y, t in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_ = self.model(x)
            logit = y_[0, y]
            # loss = torch.nn.functional.cross_entropy(y_, y)
            self.model.zero_grad()
            # loss.backward()
            logit.backward()
            grad = []
            for p in self.model.parameters():
                if p.grad is not None:
                    grad.append(p.grad.data.view(-1))
            memory.grad.append(torch.cat(grad, dim=0))

    def update_bias(self):
        q, r = torch.linalg.qr(torch.stack(memory.grad).T, 'reduced')
        self.bias = q.T

    def project(self):
        grad = []
        for p in self.model.parameters():
            if p.grad is not None:
                grad.append(p.grad.data.view(-1))
        grad = torch.cat(grad, dim=0)
        coordinate = (torch.mm(self.bias, grad.view(-1, 1))).T
        projection = torch.mm(coordinate, self.bias).view(-1)
        grad = grad - projection
        begin, end = 0, 0
        for p in self.model.parameters():
            if p.grad is not None:
                begin = end
                end += p.grad.numel()
                p.grad.data = grad[begin: end].reshape(p.grad.shape)

    def random_based_sample(self, _x, _y):
        x, y, t = [], [], []
        classes = np.unique(_y)
        for c in classes:
            mask = c == _y
            indices = np.where(mask)[0]
            indices = np.random.choice(indices, size=args.memory, replace=False)
            for index in indices:
                x.append(_x[index])
                y.append(_y[index])
                t.append(self.taskid - 1)
        return x, y, t

    def sample(self):
        trsf = self.traindata.trsf
        _x, _y = self.traindata._x, self.traindata._y
        x, y, t = self.random_based_sample(_x, _y)
        if args.dataset == 'CIFAR100':
            x = np.stack(x, axis=0)
            y = np.stack(y, axis=0)
            t = np.stack(t, axis=0)
            memory.x = x
            memory.y = y
            memory.t = t
        elif 'MNIST' in args.dataset:
            x__ = []
            for x_, y_, t_ in zip(x, y, t):
                x_ = Image.fromarray(x_.astype("uint8"))
                x_ = trsf(x_).squeeze()
                x__.append([x_, y_, t_])
            memory.x = x__
        elif 'ImageNet' in args.dataset:
            memory.x = x
            memory.y = y
            memory.t = t


class ER_PFCL(ER):
    """
    args:
        optim: SGD/DCD
        lr: learning rate of model
        decay: strength of penalize
        gamma: exponential decay
        memory: 1
        bs: 10
        mbs: 10
        omega: epochs of train LFD
        lambd: radius of LFD
        beta: learning rate of LFD
        eta: exp of LFD
    """

    def __init__(self, model, traindata, taskid):
        super(ER_PFCL, self).__init__(model, traindata, taskid)
        if taskid == 1:
            # traditional model for the first task
            return
        # path model for subsequent tasks
        self.model.init_path()
        self.pre_state = self.detach_state()
        if args.gamma < 1 and args.steps is None:
            # exponential decay of learning rate
            lr = max(args.lr * args.gamma ** (taskid - 1), 0.00005)
        else:
            lr = args.lr
        self.optimizer = torch.optim.SGD(list(self.model.path.values()), lr=lr, momentum=args.momentum,
                                         weight_decay=args.decay)
        if args.optim == 'DCD':
            self.lfd = self.lfd_train()
            self.model.init_path()
            self.optimizer = DCD(list(self.model.path.values()), lr=lr, momentum=args.momentum,
                                  weight_decay=args.decay, lfd=self.lfd)
        if args.steps is not None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.steps,
                                                                  gamma=args.gamma)
        else:
            self.scheduler = None
        self.model.train()

    def train(self):
        if self.taskid == 1:
            FineTune.train(self)
        else:
            self.path_train()
        self.construct_exemplar_set()
        self.recall()
        return self.model

    def lfd_train(self):

        if args.mode == 'sphere':
            # ************************* single sphere direction ********************* #
            epochs = int(args.omega)
            if epochs == 10:
                steps = [5, 7]
            elif epochs == 20:
                steps = [10, 15]
            elif epochs == 30:
                steps = [15, 25]
            elif epochs == 40:
                steps = [20, 30]
            else:
                raise ValueError('Only supports 10/20/30/40 epochs')
            radius = args.lambd
            ratios = [0.3, 0.5, 0.7, 0.9, 1.0]
            # self.model.eval()
            self.model.train()
            self.model.init_path()
            if 'ImageNet' in args.dataset:
                dataset = ImageNetData(memory.x, memory.y, memory.t, self.traindata.trsf)
            else:
                dataset = Memorydata(memory, self.traindata.trsf)
            loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=4)
            dim = self.model.parameters_dim()
            sphere = geoopt.manifolds.Sphere()
            sphere = geoopt.manifolds.Scaled(sphere, scale=radius)
            parameters = sphere.random_uniform(dim, device=self.device) * radius
            self.model.load_flat_path(parameters)

            optimizer = RiemannianSGD(list(self.model.path.values()), lr=args.beta, manifold=sphere,
                                      momentum=args.momentum)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1)
            progress = tqdm(range(1, len(loader) * epochs + 1), disable='slient' in args.opt or 'nni' in args.opt)
            progress.set_description('sphere')
            fast_states = []
            for ratio in ratios:
                fast_state = OrderedDict()
                for key, value in self.pre_state.items():
                    if key in self.model.path:
                        fast_state[key] = value.clone().detach() + self.model.path[key] * ratio
                    else:
                        fast_state[key] = value.clone().detach()
                fast_states.append(fast_state)
            for epoch in range(1, epochs + 1):
                for x, y, t in loader:
                    t = t + 1
                    x = x.to(self.device)
                    y = y.to(self.device)
                    loss = 0
                    for ratio, fast_state in zip(ratios, fast_states):
                        y_ = self.model(x, fast_state)
                        loss += self.criterion(y_, y, t) / ratio
                    loss = loss / len(ratios)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    n = torch.norm(self.model.flat_path()).item()
                    for index, ratio in enumerate(ratios):
                        for key, value in self.pre_state.items():
                            if key in self.model.path:
                                fast_states[index][key] = value.clone().detach() + self.model.path[key] * ratio
                    progress.set_postfix({'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'],
                                          'loss': loss.item(), 'radius': n})
                    progress.update(1)
                scheduler.step()
            progress.close()

        elif args.mode == 'multi-sphere':
            # **************************** multi-sphere direction ********************* #
            epochs = int(args.omega)
            if epochs == 10:
                steps = [5, 7]
            elif epochs == 20:
                steps = [10, 15]
            elif epochs == 30:
                steps = [15, 25]
            elif epochs == 40:
                steps = [20, 30]
            else:
                raise ValueError('Only supports 10/20/30/40 epochs')
            radius = args.lambd
            ratios = [0.3, 0.5, 0.7, 0.9, 1.0]
            scales = [r * radius for r in ratios]
            # self.model.eval()
            self.model.train()
            self.model.init_path()
            if 'ImageNet' in args.dataset:
                dataset = ImageNetData(memory.x, memory.y, memory.t, self.traindata.trsf)
            else:
                dataset = Memorydata(memory, self.traindata.trsf)
            loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=4)
            dim = 0
            manifolds = []
            for k, v in self.model.path.items():
                dim += v.numel()
                sphere = geoopt.manifolds.Sphere()
                manifolds.append((sphere, v.numel()))
            manifolds = tuple(manifolds)
            torus = geoopt.ProductManifold(*manifolds)
            parameters = torus.random_combined(dim, device=self.device)
            self.model.load_flat_path(parameters)

            from geoopt.optim import RiemannianSGD as RiemannianSGD_

            optimizer = RiemannianSGD_(list(self.model.path.values()), lr=args.beta, momentum=args.momentum)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1)
            progress = tqdm(range(1, len(loader) * epochs + 1), disable='slient' in args.opt or 'nni' in args.opt)
            progress.set_description('multi-sphere')
            fast_states = []
            for scale in scales:
                fast_state = OrderedDict()
                for key, value in self.pre_state.items():
                    if key in self.model.path:
                        l2norm = torch.norm(self.model.path[key].clone().detach(), p=2)
                        fast_state[key] = value.clone().detach() + self.model.path[key] / l2norm * scale
                    else:
                        fast_state[key] = value.clone().detach()
                fast_states.append(fast_state)
            for epoch in range(1, epochs + 1):
                for x, y, t in loader:
                    t = t + 1
                    x = x.to(self.device)
                    y = y.to(self.device)
                    loss = 0
                    for scale, fast_state in zip(scales, fast_states):
                        y_ = self.model(x, fast_state)
                        loss += self.criterion(y_, y, t) / scale
                    loss = loss / len(ratios)
                    optimizer.zero_grad()
                    loss.backward()
                    # for key, value in self.model.path.items():
                    #     torch.nn.utils.clip_grad_norm_(value, 100000000)
                    optimizer.step()
                    # norms = []
                    # for k, v in model.path.items():
                    #     norms.append(round(v.detach().norm().cpu().numpy().round(2)))
                    for index, scale in enumerate(scales):
                        for key, value in self.pre_state.items():
                            if key in self.model.path:
                                l2norm = torch.norm(self.model.path[key].clone().detach(), p=2)
                                fast_states[index][key] = value.clone().detach() + self.model.path[key] / l2norm * scale
                    progress.set_postfix({'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'],
                                          'loss': loss.item()})
                    progress.update(1)
                scheduler.step()
            progress.close()

        else:
            raise ValueError

        lfd = self.state_detach(self.model.path)

        if args.eta != 0:
            if hasattr(memory, 'lfd'):
                pre_lfd = memory.lfd
                if isinstance(lfd, dict):
                    for key, value in lfd.items():
                        lfd[key] = value * (1 - args.eta) + pre_lfd[key] * args.eta
                else:
                    lfd = lfd * (1 - args.eta) + pre_lfd * args.eta
                    lfd = torch.nn.functional.normalize(lfd, p=2, dim=0)
                memory.lfd = lfd
            else:
                memory.lfd = lfd

        self.model.init_path()
        return lfd

    def path_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt or 'nni' in args.opt)
        self.progress.set_description('train')
        fast_state = OrderedDict()
        for key, value in self.pre_state.items():
            if key in self.model.path:
                fast_state[key] = value.clone().detach() + self.model.path[key]
            else:
                fast_state[key] = value.clone().detach()
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                y_ = self.model(x, fast_state)
                loss = self.criterion(y_, y, t)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item()})
                self.progress.update(1)
                for key, value in self.model.path.items():
                    fast_state[key] = self.pre_state[key].clone().detach() + value
            if self.scheduler is not None:
                self.scheduler.step()
        self.model.load_state_dict(fast_state)
        self.progress.close()

    def recall(self):
        if self.taskid == 1 or args.recall is False:
            return
        if args.zeta != 0:
            cur_state = self.model.state_dict()
            fast_state = OrderedDict()
            for key, value in self.pre_state.items():
                fast_state[key] = self.pre_state[key] * (1 - args.zeta) + cur_state[key] * args.zeta
            self.model.load_state_dict(fast_state)
            return
        lr = args.delta
        epochs = int(args.theta)
        if epochs == 10:
            steps = [5, 7]
        elif epochs == 20:
            steps = [10, 15]
        elif epochs == 30:
            steps = [15, 25]
        elif epochs == 40:
            steps = [20, 30]
        else:
            raise ValueError('Only supports 10/20/30/40 epochs')
        if 'ImageNet' in args.dataset:
            dataset = ImageNetData(memory.x, memory.y, memory.t, self.traindata.trsf)
        else:
            dataset = Memorydata(memory, self.traindata.trsf)
        loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=4)
        factor = torch.nn.Parameter(torch.tensor(0.5).to(self.device), requires_grad=True)
        optimizer = torch.optim.SGD([factor], lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1)
        progress = tqdm(range(1, len(loader) * epochs + 1), disable='slient' in args.opt or 'nni' in args.opt)
        progress.set_description('Rcall')
        cur_state = self.model.state_dict()
        fast_state = OrderedDict()
        for key, value in self.pre_state.items():
            if key in self.model.path:
                fast_state[key] = self.pre_state[key] * (1 - factor) + cur_state[key] * factor
            else:
                fast_state[key] = cur_state[key]
        for epoch in range(1, epochs + 1):
            for x, y, t in loader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                y_ = self.model(x, fast_state)
                loss = self.criterion(y_, y, t)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(factor, 5)
                # if factor > 1:
                #     factor.data = torch.tensor(1.).to(self.device)
                # elif factor < 0:
                #     factor.data = torch.tensor(0.).to(self.device)
                optimizer.step()
                self.loss = loss.item()
                progress.set_postfix({'epoch': epoch, 'factor': round(factor.item(), 3),
                                      'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                                      'avgloss': round(self.loss, 3), 'loss': loss.item()})
                progress.update(1)
                for key, value in self.model.path.items():
                    fast_state[key] = self.pre_state[key] * (1 - factor) + cur_state[key] * factor
            scheduler.step()
        self.model.load_state_dict(fast_state)
        progress.close()

        # factor = factor.cpu().data.numpy()
        # if hasattr(memory, 'factor'):
        #     memory.factor.append(factor)
        # else:
        #     memory.__setattr__('factor', [factor])
        # if self.taskid == args.tasks:
        #     factors = np.array(memory.factor)
        #     np.save('{}.npy'.format(args.logdir+args.name+'/factors'), factors)


class iCaRL(ER):
    def __init__(self, model, traindata, taskid):
        super(iCaRL, self).__init__(model, traindata, taskid)
        assert args.tau > 0, 'Temperature should be larger than 0.'
        assert args.memory > 0
        assert 'herding' in args.opt
        if self.taskid == 1:
            return
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                         weight_decay=args.decay)
        if args.steps is not None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.steps,
                                                                  gamma=args.gamma)
        else:
            self.scheduler = None
        self.pre_minclass, self.pre_maxclass = get_minmax_class(self.taskid - 1)

    def train(self):
        self.icarl_train()
        self.construct_exemplar_set()
        pre_model = deepcopy(self.model)
        setattr(memory, 'pre_model', self.freeze_model(pre_model))
        return self.model

    def icarl_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt or 'nni' in args.opt)
        self.progress.set_description('train')
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                y_ = self.model(x)
                loss = self.criterion(y_, y, t)
                if self.taskid != 1:
                    if args.scenario == 'class':
                        y_log_ = torch.nn.functional.log_softmax(
                            y_[:, self.pre_minclass: self.pre_maxclass + 1] / args.tau, dim=1)
                        pre_output = torch.nn.functional.softmax(
                            memory.pre_model(x).detach()[:, self.pre_minclass: self.pre_maxclass + 1]
                            / args.tau, dim=1)
                        loss_div = torch.nn.functional.kl_div(y_log_, pre_output, reduction='batchmean')
                    elif args.scenario == 'domain':
                        y_log_ = torch.nn.functional.log_softmax(y_ / args.tau, dim=1)
                        pre_output = torch.nn.functional.softmax(memory.pre_model(x).detach() / args.tau, dim=1)
                        loss_div = torch.nn.functional.kl_div(y_log_, pre_output, reduction='batchmean')
                    elif args.scenario == 'task':
                        loss_div = 0
                        for t_ in range(1, self.taskid):
                            minclass, maxclass = get_minmax_class(t_)
                            y_log_ = torch.nn.functional.log_softmax(
                                y_[:, minclass: maxclass + 1] / args.tau, dim=1)
                            pre_output = torch.nn.functional.softmax(
                                memory.pre_model(x).detach()[:, minclass: maxclass + 1]
                                / args.tau, dim=1)
                            loss_div += torch.nn.functional.kl_div(y_log_, pre_output, reduction='batchmean')
                    else:
                        raise ValueError
                    loss = loss + loss_div
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item()})
                self.progress.update(1)
            if self.scheduler is not None:
                self.scheduler.step()
        self.progress.close()

    @staticmethod
    def freeze_model(model):
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model


class iCaRL_PFCL(ER_PFCL):
    def __init__(self, model, traindata, taskid):
        super(iCaRL_PFCL, self).__init__(model, traindata, taskid)
        assert args.tau > 0, 'Temperature should be larger than 0.'
        assert args.optim == 'DCD'
        assert args.memory > 0
        assert 'herding' in args.opt
        if self.taskid == 1:
            return

        self.pre_minclass, self.pre_maxclass = get_minmax_class(self.taskid - 1)

    def train(self):
        if self.taskid == 1:
            FineTune.train(self)
        else:
            self.icarl_path_train()
        self.construct_exemplar_set()
        self.recall()
        pre_model = deepcopy(self.model)
        setattr(memory, 'pre_model', self.freeze_model(pre_model))
        return self.model

    def icarl_path_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt or 'nni' in args.opt)
        self.progress.set_description('train')
        fast_state = OrderedDict()
        for key, value in self.pre_state.items():
            if key in self.model.path:
                fast_state[key] = value.clone().detach() + self.model.path[key]
            else:
                fast_state[key] = value.clone().detach()
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                y_ = self.model(x, fast_state)
                loss = self.criterion(y_, y, t)
                if self.taskid != 1:
                    if args.scenario == 'class':
                        y_log_ = torch.nn.functional.log_softmax(
                            y_[:, self.pre_minclass: self.pre_maxclass + 1] / args.tau, dim=1)
                        pre_output = torch.nn.functional.softmax(
                            memory.pre_model(x).detach()[:, self.pre_minclass: self.pre_maxclass + 1]
                            / args.tau, dim=1)
                        loss_div = torch.nn.functional.kl_div(y_log_, pre_output, reduction='batchmean')
                    elif args.scenario == 'domain':
                        y_log_ = torch.nn.functional.log_softmax(y_ / args.tau, dim=1)
                        pre_output = torch.nn.functional.softmax(memory.pre_model(x).detach() / args.tau, dim=1)
                        loss_div = torch.nn.functional.kl_div(y_log_, pre_output, reduction='batchmean')
                    elif args.scenario == 'task':
                        loss_div = 0
                        for t_ in range(1, self.taskid):
                            minclass, maxclass = get_minmax_class(t_)
                            y_log_ = torch.nn.functional.log_softmax(
                                y_[:, minclass: maxclass + 1] / args.tau, dim=1)
                            pre_output = torch.nn.functional.softmax(
                                memory.pre_model(x).detach()[:, minclass: maxclass + 1]
                                / args.tau, dim=1)
                            loss_div += torch.nn.functional.kl_div(y_log_, pre_output, reduction='batchmean')
                    else:
                        raise ValueError
                    loss = loss + loss_div
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item()})
                self.progress.update(1)
                for key, value in self.model.path.items():
                    fast_state[key] = self.pre_state[key].clone().detach() + value
            if self.scheduler is not None:
                self.scheduler.step()
        self.model.load_state_dict(fast_state)
        self.progress.close()

    @staticmethod
    def freeze_model(model):
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model


class MIR(ER):
    """
    Aljundi R, Belilovsky E, Tuytelaars T, et al.
    Online continual learning with maximal interfered retrieval[J].
    Advances in neural information processing systems, 2019, 32.
    Args:
        mcat: True
        memory: number of samples stored in memory for each class
        bs:
        mbs: number of subsamples
        zeta: top zeta interference
    """

    def __init__(self, model, traindata, taskid):
        super(MIR, self).__init__(model, traindata, taskid)
        assert args.memory > 0
        assert args.zeta > 0
        assert args.mbs >= args.zeta
        assert args.mcat is True
        self.zeta = int(args.zeta)
        if self.taskid == 1:
            return
        self.lr = args.lr
        self.tmp_state = OrderedDict()
        self.cur_state = self.model.state_dict()
        self.parameters = list(self.model.parameters())
        self.named_param = OrderedDict(self.model.named_parameters())

    def train(self):
        if self.taskid == 1:
            FineTune.train(self)
        else:
            self.mir_train()
        self.construct_exemplar_set()
        return self.model

    def mir_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt or 'nni' in args.opt)
        self.progress.set_description('train')
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                mask = t != self.taskid
                if mask.sum() > self.zeta:
                    mx, my, mt = x[mask], y[mask], t[mask]
                    cx, cy, ct = x[~mask], y[~mask], t[~mask]
                    self.model.eval()
                    y_ = self.model(x)
                    self.model.train()
                    loss = self.criterion(y_[~mask], cy, ct)
                    grad = torch.autograd.grad(loss, self.parameters)
                    index = 0
                    for key, value in self.cur_state.items():
                        if key in self.named_param:
                            self.tmp_state[key] = value.clone().detach() - self.lr * grad[index]
                            index += 1
                        else:
                            self.tmp_state[key] = value.clone().detach()
                    my_pre = y_[mask]
                    my_aft = self.model(mx, self.tmp_state)
                    loss_pre, loss_aft = [], []
                    for t_ in mt.unique():
                        mask_ = t_ == mt
                        loss_pre.append(torch.nn.functional.cross_entropy(my_pre[mask_], my[mask_], reduction="none"))
                        loss_aft.append(torch.nn.functional.cross_entropy(my_aft[mask_], my[mask_], reduction="none"))
                    loss_pre = torch.cat(loss_pre, dim=0)
                    loss_aft = torch.cat(loss_aft, dim=0)
                    score = loss_aft - loss_pre
                    index_ = score.sort(descending=True)[1][:self.zeta]
                    x = torch.cat([cx, mx[index_]], dim=0)
                    y = torch.cat([cy, my[index_]], dim=0)
                    t = torch.cat([ct, mt[index_]], dim=0)
                    y_ = self.model(x)
                    loss = self.criterion(y_, y, t)
                else:
                    y_ = self.model(x)
                    loss = self.criterion(y_, y, t)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item()})
                self.progress.update(1)
            if self.scheduler is not None:
                self.scheduler.step()
        self.progress.close()
        return self.model


class MIR_PFCL(ER_PFCL):
    def __init__(self, model, traindata, taskid):
        super(MIR_PFCL, self).__init__(model, traindata, taskid)
        assert args.optim == 'DCD'
        assert args.memory > 0
        assert args.zeta > 0
        assert args.mbs >= args.zeta
        assert args.mcat is True
        if self.taskid == 1:
            return
        self.lr = args.lr
        self.zeta = int(args.zeta)
        self.tmp_state = OrderedDict()
        self.parameters = list(self.model.path.values())

    def train(self):
        if self.taskid == 1:
            FineTune.train(self)
        else:
            self.mir_path_train()
        self.construct_exemplar_set()
        self.recall()
        return self.model

    def mir_path_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt or 'nni' in args.opt)
        self.progress.set_description('train')
        fast_state = OrderedDict()
        for key, value in self.pre_state.items():
            if key in self.model.path:
                fast_state[key] = value.clone().detach() + self.model.path[key]
            else:
                fast_state[key] = value.clone().detach()
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                mask = t != self.taskid
                if mask.sum() > self.zeta:
                    mx, my, mt = x[mask], y[mask], t[mask]
                    cx, cy, ct = x[~mask], y[~mask], t[~mask]
                    self.model.eval()
                    y_ = self.model(x, fast_state)
                    self.model.train()
                    loss = self.criterion(y_[~mask], cy, ct)
                    grad = torch.autograd.grad(loss, self.parameters)  # TODO
                    index = 0
                    for key, value in self.pre_state.items():
                        if key in self.model.path:
                            self.tmp_state[key] = value.clone().detach() + self.model.path[key].clone().detach() - \
                                                  self.lr * grad[index]
                            index += 1
                        else:
                            self.tmp_state[key] = fast_state[key].clone().detach()
                    my_pre = y_[mask]
                    my_aft = self.model(mx, self.tmp_state)
                    loss_pre, loss_aft = [], []
                    for t_ in mt.unique():
                        mask_ = t_ == mt
                        loss_pre.append(torch.nn.functional.cross_entropy(my_pre[mask_], my[mask_], reduction="none"))
                        loss_aft.append(torch.nn.functional.cross_entropy(my_aft[mask_], my[mask_], reduction="none"))
                    loss_pre = torch.cat(loss_pre, dim=0)
                    loss_aft = torch.cat(loss_aft, dim=0)
                    score = loss_aft - loss_pre
                    index_ = score.sort(descending=True)[1][:self.zeta]
                    x = torch.cat([cx, mx[index_]], dim=0)
                    y = torch.cat([cy, my[index_]], dim=0)
                    t = torch.cat([ct, mt[index_]], dim=0)
                    y_ = self.model(x, fast_state)
                    loss = self.criterion(y_, y, t)
                else:
                    y_ = self.model(x, fast_state)
                    loss = self.criterion(y_, y, t)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item()})
                self.progress.update(1)
                for key, value in self.model.path.items():
                    fast_state[key] = self.pre_state[key].clone().detach() + value
            if self.scheduler is not None:
                self.scheduler.step()
        self.model.load_state_dict(fast_state)
        self.progress.close()
