import torch
import torch.nn as nn
from collections import OrderedDict


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0.):
        super(BasicBlock, self).__init__()
        self.residual_function1 = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
            nn.Dropout(p=dropout)
        )

        if downsample is None:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = downsample
        self.residual_function2 = nn.Sequential(
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.residual_function1(x)
        out += self.shortcut(x)
        out = self.residual_function2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0.):
        super(Bottleneck, self).__init__()
        self.residual_function1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion),
            nn.Dropout(p=dropout)
        )
        if downsample is None:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = downsample
        self.residual_function2 = nn.Sequential(
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.residual_function1(x)
        out += self.shortcut(x)
        out = self.residual_function2(out)
        return out


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.pointer = -1
        self.counter_bn = 0
        self.path = OrderedDict()

    def init_path(self):
        for key, value in self.named_parameters():
            self.path[key] = torch.nn.Parameter(0.0 * torch.ones_like(value), requires_grad=True)

    def parameters_dim(self):
        dim = 0
        for value in self.parameters():
            dim += value.numel()
        return dim

    def load_flat_path(self, p):
        start = 0
        stop = 0
        for key, value in self.path.items():
            stop = stop + value.numel()
            p_ = p[start:stop].reshape(value.shape)
            value.data = p_.data
            start = stop

    def flat_path(self):
        p = []
        for key, value in self.path.items():
            p.append(value.view(1, -1))
        p = torch.cat(p, dim=1)
        return p

    def load_grad(self, grad: OrderedDict):
        for k, v in self.named_parameters():
            v.grad = grad[k]

    def load_parameter(self, p):
        for key, value in self.named_parameters():
            value.data = p[key].data

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_with_parameters(self, modules, parameters, x):
        for name, module in modules.named_children():
            # ****************** Sequential ******************* #
            if type(module) is torch.nn.Sequential:
                x = self.forward_with_parameters(module, parameters, x)
            # ****************** Conv2d ******************* #
            elif type(module) is torch.nn.Conv2d:
                self.pointer += 1
                weight = parameters[self.pointer][1]
                if module.bias is None:
                    bias = None
                else:
                    self.pointer += 1
                    bias = parameters[self.pointer][1]
                x = torch.nn.functional.conv2d(x, weight=weight, bias=bias, stride=module.stride,
                                               padding=module.padding, dilation=module.dilation, groups=module.groups)
            # ****************** BatchNorm2d ******************* #
            elif type(module) is torch.nn.BatchNorm2d:
                # ****************** affine ***************** #
                if module.affine is False:
                    weight = None
                    bias = None
                else:
                    self.pointer += 1
                    weight = parameters[self.pointer][1]
                    if module.bias is None:
                        bias = None
                    else:
                        self.pointer += 1
                        bias = parameters[self.pointer][1]
                # ***************** statistics **************** #
                if ('running_mean' in parameters[self.pointer + 1][0]) and \
                        ('running_var' in parameters[self.pointer + 2][0]):
                    # meta-SGD, model's mean and var is static,
                    # but, parameters' mean and var is changing.
                    self.pointer += 1
                    running_mean = parameters[self.pointer][1]
                    self.pointer += 1
                    running_var = parameters[self.pointer][1]
                    # point to num_batches_tracked
                    self.pointer += 1
                    parameters[self.pointer][1].data += 1
                    track_running_stats = True
                else:
                    # MAML, model's mean and var is changing.
                    state = list(self.state_dict().items())
                    if 'running_mean' in state[self.pointer + self.counter_bn + 1][0] and 'running_var' in \
                            state[self.pointer + self.counter_bn + 2][0]:
                        self.counter_bn += 1
                        running_mean = state[self.pointer + self.counter_bn][1]
                        self.counter_bn += 1
                        running_var = state[self.pointer + self.counter_bn][1]
                        self.counter_bn += 1
                        state[self.pointer + self.counter_bn][1].data += 1
                        track_running_stats = True
                    else:
                        running_mean = None
                        running_var = None
                        track_running_stats = False
                x = torch.nn.functional.batch_norm(x, running_mean=running_mean, running_var=running_var,
                                                   weight=weight, bias=bias,
                                                   training=module.training or not track_running_stats,
                                                   momentum=module.momentum, eps=module.eps)
            # ****************** ReLU ******************* #
            elif type(module) is torch.nn.ReLU:
                x = torch.nn.functional.relu(x, inplace=module.inplace)
            # ****************** Linear ******************* #
            elif type(module) is torch.nn.Linear:
                self.pointer += 1
                weight = parameters[self.pointer][1]
                self.pointer += 1
                bias = parameters[self.pointer][1]
                if x.shape[2:4] == (1, 1):
                    x = x.view(x.shape[0], -1)
                x = torch.nn.functional.linear(x, weight=weight, bias=bias)
            # ****************** AdaptiveAvgPool2d ******************* #
            elif type(module) is torch.nn.Identity:
                pass
            # ****************** AdaptiveAvgPool2d ******************* #
            elif type(module) is torch.nn.AvgPool2d:
                x = torch.nn.functional.avg_pool2d(x, module.kernel_size)
            # ********************* MaxPool2d ************************ #
            elif type(module) is torch.nn.MaxPool2d:
                x = torch.nn.functional.max_pool2d(x, module.kernel_size, module.stride, module.padding,
                                                   module.dilation, module.ceil_mode, module.return_indices)
            # *********************** Dropout ************************ #
            elif type(module) is torch.nn.Dropout:
                p = module.p
                inplace = module.inplace
                training = module.training
                x = torch.nn.functional.dropout(x, p=p, training=training, inplace=inplace)
            # ****************** BasicBlock ******************* #
            elif type(module) is BasicBlock:
                x1 = self.forward_with_parameters(module.residual_function1, parameters, x)
                x2 = self.forward_with_parameters(module.shortcut, parameters, x)
                x = x1 + x2
                x = self.forward_with_parameters(module.residual_function2, parameters, x)
        return x


class ResNet(BaseNet):

    def __init__(self, args, block, layers, num_classes=100):
        self.inplanes = 16
        dropout = args.dropout
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 16, layers[0], dropout=dropout)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dropout=dropout)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dropout=0.):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
                nn.Dropout(p=dropout)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout=dropout))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x, parameters=None):
        if parameters is None:
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        else:
            self.pointer = -1
            self.counter_bn = 0
            assert type(parameters) is OrderedDict
            parameters = list(parameters.items())
            out = self.forward_with_parameters(self, parameters, x)
        return out


class ResNet20_Class(ResNet):
    def __init__(self, args, num_classes=100):
        n = 3
        super(ResNet20_Class, self).__init__(args, BasicBlock, [n, n, n], num_classes=num_classes)


class ResNet32_Class(ResNet):
    def __init__(self, args, num_classes=100):
        n = 5
        super(ResNet32_Class, self).__init__(args, BasicBlock, [n, n, n], num_classes=num_classes)


class ResNet56_Class(ResNet):
    def __init__(self, args, num_classes=100):
        n = 9
        super(ResNet56_Class, self).__init__(args, Bottleneck, [n, n, n], num_classes=num_classes)


if __name__ == "__main__":
    def detach_parameters(model):
        parameters = OrderedDict()
        for key, value in model.named_parameters():
            parameters[key] = value.clone().detach()
        return parameters

    def detach_state(model):
        state = OrderedDict()
        for key, value in model.state_dict().items():
            state[key] = value.clone().detach()
        return state

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', default=0)
    parser.add_argument('--tasks', default=20)
    args = parser.parse_args()
    model = ResNet32_Class(args)
    parameters = detach_parameters(model)
    state = detach_state(model)
    x = torch.randn(2, 3, 32, 32)
    # x = torch.randn(2, 784)
    y_ = model(x, parameters=state)
    y = model(x)
    print(y - y_)
    a = 1
