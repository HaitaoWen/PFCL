import os
import sys
import nni
import time
import torch
import random
import shutil
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
from parser import args
from datetime import datetime
from yacs.config import CfgNode
from continuum import ClassIncremental
from continuum import Permutations, Rotations
from torchvision.transforms import transforms
from models.cifar_class import ResNet32_Class
from models.model_task import ResNet18_Task, MLP
from models.imagenet_class import ResNet18_Class
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TorchVisionFunc
from continuum.datasets import MNIST, CIFAR100, TinyImageNet200, ImageNet100, ImageNet1000
warnings.filterwarnings("ignore", category=RuntimeWarning)


# **************** input ****************** #
datainfo = {
    'perMNIST': {
        'classnum': 10
    },
    'rotMNIST': {
        'classnum': 10
    },
    'CIFAR100': {
        'classnum': 100
    },
    'TinyImageNet': {
        'classnum': 200
    },
    'miniImageNet': {
        'classnum': 100
    },
    'ImageNet': {
        'classnum': 1000
    }
}


Transforms = {
        'perMNIST':
            {'train': [transforms.ToTensor()],
             'eval': [transforms.ToTensor()]},

        'rotMNIST':
            {'train': [transforms.ToTensor()],
             'eval': [transforms.ToTensor()]},

        'CIFAR100':
            {'train': [
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=63 / 255),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                                     std=[0.2673, 0.2564, 0.2762])],
                'eval': [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                                         std=[0.2673, 0.2564, 0.2762])]},

        'TinyImageNet':
            {'train': [
                # transforms.ToPILImage(),
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=63 / 255),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])],
                'eval': [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                         std=[0.2302, 0.2265, 0.2262])]},

        'miniImageNet':
            {'train': [
                transforms.Resize((224, 224)),
                # transforms.ToPILImage(),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=63 / 255),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
                'eval': [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]},

        'ImageNet':
            {'train': [
                transforms.Resize((224, 224)),
                # transforms.ToPILImage(),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=63 / 255),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
                'eval': [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]}
}


def load_scenario():

    def arrange_increment():
        if args.init is None:
            increment_ = datainfo[args.dataset]['classnum'] / args.tasks
        else:
            assert isinstance(args.init, int)
            left = datainfo[args.dataset]['classnum'] - args.init
            increment_ = left / (args.tasks - 1)
        if not increment_.is_integer():
            raise ValueError('number of classes {} should be divided by number of tasks {}'.format(
                datainfo[args.dataset]['classnum'] if args.init is None else left,
                args.tasks if args.init is None else args.tasks-1))
        else:
            increment_ = int(increment_)
        return increment_

    # def get_order_map(scenario_train_):
    #     class_order_ = scenario_train_.class_order.tolist()
    #     labelmap_ = []
    #     for i in range(len(class_order_)):
    #         labelmap_.append(class_order_.index(i))
    #     return class_order_, labelmap_

    if args.dataset == 'perMNIST':
        scenario_train = Permutations(
            MNIST(data_path=args.datadir, download=True, train=True),
            nb_tasks=args.tasks,
            base_transformations=Transforms[args.dataset]['train'],
            seed=args.seed,
            shared_label_space=True
        )
        scenario_eval = Permutations(
            MNIST(data_path=args.datadir, download=True, train=False),
            nb_tasks=args.tasks,
            base_transformations=Transforms[args.dataset]['eval'],
            seed=args.seed,
            shared_label_space=True
        )
        args.increments = [datainfo[args.dataset]['classnum']] * args.tasks
        # args.class_order = [i for i in range(datainfo[args.dataset]['classnum']*args.tasks)]
        # args.labelmap = args.class_order
    elif args.dataset == 'rotMNIST':
        trsf = []
        degrees = []
        # MC-SGD implementation
        per_task_rotation = 9
        for task_id in range(args.tasks):
            degree = task_id * per_task_rotation
            degrees.append(degree)
            # trsf.append([RotationTransform(degree)])
        # Stable-SGD implementation
        # per_task_rotation = 10
        # for task_id in range(args.tasks):
        #     rotation_degree = task_id * per_task_rotation
        #     rotation_degree -= (np.random.random() * per_task_rotation)
        #     degrees.append(int(rotation_degree))
        #     trsf.append([RotationTransform(rotation_degree)])
        scenario_train = Rotations(
            MNIST(data_path=args.datadir, download=True, train=True),
            nb_tasks=args.tasks,
            base_transformations=Transforms[args.dataset]['train'],
            list_degrees=degrees,
            shared_label_space=True
        )
        scenario_eval = Rotations(
            MNIST(data_path=args.datadir, download=True, train=False),
            nb_tasks=args.tasks,
            base_transformations=Transforms[args.dataset]['eval'],
            list_degrees=degrees,
            shared_label_space=True
        )
        # scenario_train.inc_trsf = trsf
        # scenario_eval.inc_trsf = trsf
        args.increments = [datainfo[args.dataset]['classnum']] * args.tasks
        # args.class_order = [i for i in range(datainfo[args.dataset]['classnum']*args.tasks)]
        # args.labelmap = args.class_order
    elif args.dataset == 'CIFAR100':
        increment = arrange_increment()
        if args.order is not None:
            file = open('orders.yaml')
            order = CfgNode.load_cfg(file)[args.dataset]['order'][args.order]
        else:
            order = None
        scenario_train = ClassIncremental(
            CIFAR100(data_path=args.datadir, download=True & ('slient' not in args.opt), train=True),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['train'],
            class_order=order
        )
        scenario_eval = ClassIncremental(
            CIFAR100(data_path=args.datadir, download=True & ('slient' not in args.opt), train=False),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['eval'],
            class_order=order
        )
        args.increments = scenario_train.increments
        # args.class_order, args.labelmap = get_order_map(scenario_train)
    elif args.dataset == 'TinyImageNet':
        increment = arrange_increment()
        scenario_train = ClassIncremental(
            TinyImageNet200(data_path=args.datadir, download=True, train=True),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['train']
        )
        scenario_eval = ClassIncremental(
            TinyImageNet200(data_path=args.datadir, download=True, train=False),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['eval']
        )
        args.increments = scenario_train.increments
        # args.class_order, args.labelmap = get_order_map(scenario_train)
    elif args.dataset == 'miniImageNet':
        increment = arrange_increment()
        scenario_train = ClassIncremental(
            ImageNet100(data_path=args.datadir+'/ImageNet/', download=True, train=True),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['train']
        )
        scenario_eval = ClassIncremental(
            ImageNet100(data_path=args.datadir+'/ImageNet/', download=True, train=False),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['eval']
        )
        args.increments = scenario_train.increments
        # args.class_order, args.labelmap = get_order_map(scenario_train)
    elif args.dataset == 'ImageNet':
        increment = arrange_increment()
        scenario_train = ClassIncremental(
            ImageNet1000(data_path=args.datadir+'/ImageNet/', download=True, train=True),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['train']
        )
        scenario_eval = ClassIncremental(
            ImageNet1000(data_path=args.datadir+'/ImageNet/', download=True, train=False),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['eval']
        )
        args.increments = scenario_train.increments
        # args.class_order, args.labelmap = get_order_map(scenario_train)
    else:
        raise ValueError('{} data is not supported!'.format(args.dataset))

    # ***************** MultiTask ***************** #
    if args.scheme == 'MultiTask':
        scenario_train = [scenario_train]
    return scenario_train, scenario_eval


class RotationTransform:
    """
    Rotation transforms for the images in `Rotation MNIST` dataset.
    """
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TorchVisionFunc.rotate(x, self.angle, fill=(0,))


class CommonData(Dataset):
    def __init__(self, traindata, transform):
        self.traindata = traindata
        self.transform = transform

    def __getitem__(self, index):
        x, y, t = self.traindata[index]
        if self.transform is not None:
            x = Image.fromarray(x.astype("uint8"))
            x = self.transform(x)
        return x, y, t

    def __len__(self):
        return len(self.traindata)


class CIFARData(Dataset):
    def __init__(self, traindata, memory):
        self.transform = traindata.trsf
        _x, _y, _t = traindata._x, traindata._y, traindata._t
        self.x = np.concatenate((_x, memory.x), axis=0)
        self.y = np.concatenate((_y, memory.y), axis=0)
        self.t = np.concatenate((_t, memory.t), axis=0)

    def __getitem__(self, index):
        x, y, t = self.x[index], self.y[index], self.t[index]
        if self.transform is not None:
            x = Image.fromarray(x.astype("uint8"))
            x = self.transform(x)
        return x, y, t

    def __len__(self):
        return len(self.x)


class MNISTData(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Images(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            if 'ImageNet' in args.dataset:
                image = Image.open(image).convert("RGB")
            else:
                image = Image.fromarray(image.astype("uint8"))
            image = self.transform(image)
        return image, index

    def __len__(self):
        return len(self.images)


class Memorydata(Dataset):
    def __init__(self, mem, transform):
        if args.dataset == 'CIFAR100':
            self.transform = transform
            self.x, self.y, self.t = mem.x, mem.y, mem.t
        elif 'MNIST' in args.dataset:
            self.transform = None
            self.x, self.y, self.t = zip(*mem.x)
        else:
            raise ValueError

    def __getitem__(self, index):
        x, y, t = self.x[index], self.y[index], self.t[index]
        if self.transform is not None:
            x = Image.fromarray(x.astype("uint8"))
            x = self.transform(x)
        return x, y, t

    def __len__(self):
        return len(self.x)


class ImageNetData(Dataset):
    def __init__(self, x, y, t, transform):
        self.transform = transform
        self.x, self.y, self.t = x, y, t

    def __getitem__(self, index):
        x, y, t = self.x[index], self.y[index], self.t[index]
        image = Image.open(x).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, y, t

    def __len__(self):
        return len(self.x)


# **************** output ****************** #
def printlog(string: str, printed=True, delay=0):
    if printed is True and 'slient' not in args.opt:
        print(string)
        if delay > 0:
            time.sleep(delay)
    if 'nni' in args.opt:
        return
    if not os.path.exists(args.logdir+args.name):
        os.makedirs(args.logdir+args.name)
    txt = open(args.logdir+args.name+'/log.txt', 'a+')
    txt.write(string + '\n')
    txt.close()


def print_args():
    printlog('--------------args--------------')
    for k in list(vars(args).keys()):
        printlog('%s: %s' % (k, vars(args)[k]))
    printlog('--------------args--------------')


def backup():
    if 'slient' in args.opt or 'nni' in args.opt:
        return
    files = [f for f in os.listdir() if '.py' in f]
    files_dir = os.path.join(args.logdir+args.name, 'program')
    if os.path.exists(files_dir) is True and not nni_params:
        shutil.rmtree(files_dir)
    if os.path.exists(files_dir) is False:
        os.makedirs(files_dir)
        for name in files:
            os.system('cp {} {}'.format(name, files_dir))


def save_model(model, taskid):
    state = model.state_dict()
    dir_ = os.path.join(args.logdir+args.name, 'pkl')
    if os.path.exists(dir_) is False:
        os.makedirs(dir_)
    torch.save(state, os.path.join(dir_, 'task{}.pkl'.format(taskid)))


def print_log_metrics(taskid):
    Acc = AccMatrix[taskid-1, :taskid].mean().round(3)
    forget = 0
    if taskid > 1:
        subMatrix = AccMatrix[:taskid-1, :taskid-1]
        maxAcc = np.max(subMatrix, axis=0)
        curAcc = AccMatrix[taskid-1, :taskid-1]
        forget = np.mean(maxAcc - curAcc).round(3)
    Accs = []
    for i in range(taskid):
        Accs.append(AccMatrix[i, :i+1].mean().round(3))
    AIAcc = np.mean(Accs).round(3)
    info = ''
    for t in range(taskid):
        info += 'task{}:{} '.format(t+1, round(AccMatrix[taskid-1, t], 3))
    info = info[:-1] + '\n'
    info += 'Acc  :{} forget:{} AIAcc:{}'.format(Acc, forget, AIAcc)
    printlog(info)
    if 'nni' in args.opt:
        nni.report_intermediate_result({'default': Acc, 'forget': forget})
        if taskid == args.tasks:
            nni.report_final_result({'default': Acc, 'forget': forget})
    else:
        np.savetxt(args.logdir + args.name + '/AccMatrix.csv', AccMatrix, fmt='%.3f')
    if 'slient' in args.opt or 'nni' in args.opt:
        # block tensorboard
        return
    TBWriter.add_scalar('Acc', Acc, taskid)
    TBWriter.add_scalar('Forget', forget, taskid)
    TBWriter.add_scalar('AIAcc', AIAcc, taskid)
    for t in range(args.tasks):
        TBWriter.add_scalar('Task{}'.format(t+1), AccMatrix[taskid-1, t], taskid)


def summary(logdir: str):
    Accs, Forgets, AIAs = [], [], []
    exps = os.listdir(logdir)
    for exp in exps:
        path = os.path.join(logdir, exp, 'AccMatrix.csv')
        if not os.path.exists(path) or 'search' in path:
            continue
        Matrix = np.loadtxt(path)
        Acc = Matrix[-1, :].mean().round(3)
        lastAcc = Matrix[-1, :-1]
        subMatrix = Matrix[:-1, :-1]
        forget = (np.max(subMatrix, axis=0) - lastAcc).mean().round(3)
        Accs_ = []
        for i in range(Matrix.shape[0]):
            Accs_.append(Matrix[i, :i+1].mean().round(3))
        Accs.append(Acc)
        Forgets.append(forget)
        AIAs.append(np.mean(Accs_).round(3))
    meanAcc = (np.mean(Accs) * 100).round(1)
    devAcc = (np.std(Accs) * 100).round(2)
    meanForget = np.mean(Forgets).round(2)
    devForget = np.std(Forgets).round(2)
    meanAIA = (np.mean(AIAs) * 100).round(1)
    devAIA = (np.std(AIAs) * 100).round(2)
    info = '************************************************************\n'
    info += logdir + '\n{} experiments are summaried as follows:\n'.format(len(Accs))
    info += 'Acc(%):{}(+-{}) Forget:{}(+-{}) AIA(%):{}(+-{})\n'.format(meanAcc, devAcc, meanForget, devForget, meanAIA,
                                                                       devAIA)
    info += '************************************************************'
    print(info)
    with open(os.path.join(logdir, 'summary.txt'), 'w') as f:
        f.write(info)


# **************** training ******************** #
class Memory:
    def __init__(self):
        self.x = None
        self.y = None
        self.t = None


def build_model():
    if 'MNIST' in args.dataset:
        model = MLP(args, hiddens=256)
    elif 'CIFAR' in args.dataset:
        if args.scenario == 'class':
            model = ResNet32_Class(args, num_classes=100)
        else:
            model = ResNet18_Task(args, num_classes=100)
    elif 'ImageNet' in args.dataset:
        num_classes = datainfo[args.dataset]['classnum']
        model = ResNet18_Class(args, num_classes=num_classes)
    else:
        raise ValueError
    model.init_parameters()
    return model


def init_state():
    global memory, AccMatrix
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    memory = Memory()
    AccMatrix = np.zeros((args.tasks, args.tasks))


def get_minmax_class(task):
    if (args.scenario == 'task') or (args.scenario == 'domain'):
        # task: id is used to select the corresponding classifier,
        # thus, task_id is known to classifier.
        # domain: id is used to minus the shift of ground truth labels,
        # however, classifier don't know the task_id.
        # for permuted/rotated MNIST, minclass and maxclass is not correct,
        # but these values are not used in activate_head() function.
        minclass = 0
        for i in args.increments[:task-1]:
            minclass += i
        maxclass = minclass + args.increments[task-1] - 1
    elif args.scenario == 'class':
        minclass = 0
        maxclass = 0
        for i in args.increments[:task]:
            maxclass += i
        maxclass -= 1
    else:
        raise ValueError('please choose a scenario from [\'task\', \'domain\', \'class\']')
    return minclass, maxclass


def activate_head(minclass, maxclass, prediction, target):
    target_ = target.clone()
    prediction_ = prediction.clone()
    if args.scenario == 'task':
        prediction_ = prediction_[:, minclass: maxclass+1]
        target_ = target_ - minclass
        # if minclass > 0:
        #     prediction_[:, :minclass].data.fill_(-10e10)
        # if maxclass < prediction_.size(1):
        #     prediction_[:, maxclass+1:prediction_.size(1)].data.fill_(-10e10)
        return prediction_, target_
    elif args.scenario == 'class':
        prediction_ = prediction_[:, 0: maxclass+1]
        return prediction_, target_
    elif args.scenario == 'domain':
        if args.dataset in ('perMNIST', 'rotMNIST'):
            return prediction_, target_
        else:
            target_ = target_ - minclass
            return prediction_, target_


# **************** eval ******************** #
def remap_label(y_, minclass):
    y = y_.clone()
    if args.scenario == 'task':
        y = y + minclass
        return y
    elif args.scenario == 'class':
        return y
    elif args.scenario == 'domain':
        if args.dataset in ('perMNIST', 'rotMNIST'):
            # TODO, or other muti-domain dataset
            return y
        else:
            # TODO, constructed by regular dataset (e.g. CIFAR100)
            y = y + minclass
            return y


def evaluate(model, scenario, taskid):
    if nni_params:
        device = torch.device('cuda')
    else:
        device = torch.device('cuda', args.gpuid)
    model = model.to(device)
    model.eval()
    time.sleep(1)
    if args.scheme == 'MultiTask':
        taskid = args.tasks
    progress = tqdm(range(1, taskid + 1), disable='slient' in args.opt or 'nni' in args.opt)
    progress.set_description('eval ')
    for t in progress:
        evaldata = scenario[t-1]
        evalloader = DataLoader(evaldata, args.bs, shuffle=False, num_workers=8)
        if args.scenario == 'class':
            minclass, maxclass = get_minmax_class(taskid)
        else:
            minclass, maxclass = get_minmax_class(t)
        targets, predicts = [], []
        with torch.no_grad():
            for x, y, _ in evalloader:
                x = x.to(device)
                y = y.to(device)
                y_ = model(x)
                y_, _ = activate_head(minclass, maxclass, y_, y)
                y_ = y_.topk(k=1, dim=1)[1]
                y_ = remap_label(y_, minclass)
                targets.append(y)
                predicts.append(y_)
        targets = torch.cat(targets, dim=0).unsqueeze(dim=1)
        predicts = torch.cat(predicts, dim=0)
        top1_acc = targets.eq(predicts[:, :1]).sum().item() / targets.shape[0]
        AccMatrix[taskid-1, t-1] = top1_acc
    progress.close()
    # if 'save_model' in args.opt:
    #     save_model(model, taskid)
    print_log_metrics(taskid)
    time.sleep(1)


# nni search
nni_params = nni.get_next_parameter()
if nni_params:
    args.__init__(**nni_params)
    args.opt.append('nni')
# else:
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
if 'summary' in args.opt:
    summary(args.logdir + args.name)
    sys.exit(0)
# global values
memory = Memory()
AccMatrix = np.zeros((args.tasks, args.tasks))
if not ('slient' in args.opt or 'nni' in args.opt):
    current_time = datetime.now().strftime('/%b%d_%H-%M-%S')
    args.name = args.name + current_time
    TBWriter = SummaryWriter(log_dir=args.logdir + args.name, flush_secs=20)
