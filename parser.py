import argparse

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--dataset', default='CIFAR100', type=str, choices=['perMNIST', 'rotMNIST', 'CIFAR100',
                                                                        'miniImageNet'])
parser.add_argument('--datadir', default='data', type=str, help='root of dataset')
parser.add_argument('--scenario', default='task', type=str, choices=['task', 'domain', 'class'])
parser.add_argument('--order', default=None, type=int, help='index of classes order in .yaml file')
parser.add_argument('--init', default=None, type=int, help='number of classes in the first task')
parser.add_argument('--tasks', default=20, type=int, help='number of tasks needed to learn')
parser.add_argument('--end', default=None, type=int, help='early end point of tasks')
# General setting
parser.add_argument('--scheme', default='FineTune', type=str, help='')
parser.add_argument('--dropout', default=0., type=float)
parser.add_argument('--optim', default='SGD', type=str)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--decay', default=0.0, type=float)
parser.add_argument('--momentum', default=0.7, type=float)
parser.add_argument('--steps', default=None, nargs='+', type=int)
parser.add_argument('--gamma', default=1., type=float)
parser.add_argument('--sigma', default=0., type=float)
parser.add_argument('--memory', default=0, type=int)
parser.add_argument('--mcat', action='store_true')
parser.add_argument('--bs', default=10, type=int)
parser.add_argument('--mbs', default=10, type=int)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--omega', default=0., type=float)
parser.add_argument('--lambd', default=0., type=float)
parser.add_argument('--alpha', default=0., type=float)
parser.add_argument('--beta', default=0., type=float)
parser.add_argument('--tau', default=0., type=float)
parser.add_argument('--eta', default=0., type=float)
parser.add_argument('--recall', action='store_true')
parser.add_argument('--theta', default=0., type=float)
parser.add_argument('--delta', default=0., type=float)
parser.add_argument('--zeta', default=0., type=float)
parser.add_argument('--mode', default='sphere', type=str)

# Device
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--seed', default=1234, type=int)
# Output
parser.add_argument('--logdir', default='log/', type=str)
parser.add_argument('--name', default='test', type=str)
parser.add_argument('--opt', default=[], nargs='+', help="choose modes from ['summary', 'slient', "
                                                         "'save_info', 'herding', 'A-GEM']")
parser.add_argument('--grid', default='', type=str)
args = parser.parse_args()
a = 1
