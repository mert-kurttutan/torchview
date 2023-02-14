# type: ignore
# pylint: skip-file
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.checkpoint as cp


class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return {
            k: v.to_dict() if isinstance(v, CfgNode)
            else v for k, v in self.__dict__.items()
        }

    def merge_from_dict(self, d):
        self.__dict__.update(d)


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class BottleneckUnit(nn.Module):
    def __init__(
        self, in_channel, growth_rate, expansion,
        p_dropout, activation, efficient=False
    ):
        super(BottleneckUnit, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.activation = activation
        self.conv1 = nn.Conv2d(
            in_channel, expansion * growth_rate,
            kernel_size=1, stride=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(expansion * growth_rate)

        self.conv2 = nn.Conv2d(
            expansion * growth_rate, growth_rate,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.p_dropout = p_dropout
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.bn1, self.activation, self.conv1)
        if (
            self.efficient and
            any(prev_feature.requires_grad for prev_feature in prev_features)
        ):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.activation(self.bn2(bottleneck_output)))
        if self.p_dropout > 0:
            new_features = F.dropout(
                new_features, p=self.p_dropout, training=self.training
            )
        return new_features


class _InitBlock(nn.Module):

    def __init__(self, in_channel, out_channel, small_inputs, activation):
        super(_InitBlock, self).__init__()

        self.activation = activation
        self.small_inputs = small_inputs
        if small_inputs:
            self.conv = nn.Conv2d(
                in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            )

        else:
            self.conv = nn.Conv2d(
                in_channel, out_channel, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.bn = nn.BatchNorm2d(out_channel)
            self.pool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, ceil_mode=False
            )

    def forward(self, x):

        out = self.conv(x)
        if not self.small_inputs:
            out = self.pool(self.activation(self.bn(out)))

        return out


class _TransitionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation):
        super(_TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        self.activation = activation
        self.conv = nn.Conv2d(
            in_channel, out_channel,
            kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.activation(self.bn(x))
        out = self.pool(self.conv(out))

        return out


class _DenseBlock(nn.Module):
    def __init__(
        self, dense_unit, n_unit, in_channel,
        expansion, growth_rate, p_dropout, activation, efficient=False
    ):
        super(_DenseBlock, self).__init__()
        for i in range(n_unit):
            layer = dense_unit(
                in_channel=in_channel + i * growth_rate,
                growth_rate=growth_rate,
                expansion=expansion,
                p_dropout=p_dropout,
                efficient=efficient,
                activation=activation,
            )
            self.add_module('dense_unit%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for _, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        n_channel (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        n_class (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32.
        Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing.
        Much more memory efficient, but slower.
    """

    @staticmethod
    def get_activation(name: str):

        activation_map = {
            "relu": F.relu,
            "elu": F.elu,
            "gelu": F.gelu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }

        return activation_map[name]

    @staticmethod
    def get_default_config():
        C = CfgNode()
        # either model_type or (n_layer, n_head, n_embd)
        # must be given in the config
        C.model_type = 'densenet'
        C.growth_rate = 24
        C.n_channel = C.growth_rate * 2
        C.n_channel = 32
        C.img_channel = 3
        C.activation = "relu"
        C.p_dropout = 0.16
        C.compression = 0.5
        C.expansion = 4
        C.fc_pdrop = 0.16
        C.small_inputs = True
        C.efficient = True
        C.n_class = 10
        C.n_blocks = [5, 5, 5]

        return C

    def __init__(
        self,
        config: CfgNode,
        unit_module=BottleneckUnit,
    ):

        self.activation = (
            F.relu if config.activation is None
            else self.get_activation(config.activation)
        )

        super(DenseNet, self).__init__()
        assert 0 < config.compression <= 1, (
            'compression of densenet should be between 0 and 1'
        )

        self.features = nn.Sequential()

        # First convolution
        self.features.add_module(
            "init_block",
            _InitBlock(
                config.img_channel, config.n_channel,
                config.small_inputs, self.activation
            )
        )

        # Each denseblock
        n_feature = config.n_channel
        for i, n_unit in enumerate(config.n_blocks):
            block = _DenseBlock(
                dense_unit=unit_module,
                n_unit=n_unit,
                in_channel=n_feature,
                expansion=config.expansion,
                growth_rate=config.growth_rate,
                p_dropout=config.p_dropout,
                efficient=config.efficient,
                activation=self.activation,
            )
            self.features.add_module('dense_block_%d' % (i + 1), block)
            n_feature = n_feature + n_unit * config.growth_rate
            if i != len(config.n_blocks) - 1:
                trans = _TransitionBlock(
                    in_channel=n_feature,
                    out_channel=int(n_feature * config.compression),
                    activation=self.activation
                )
                self.features.add_module('transition_%d' % (i + 1), trans)
                n_feature = int(n_feature * config.compression)

        # Final batch norm
        self.bn_final = nn.BatchNorm2d(n_feature)

        # fully connect / classifer layer
        self.fc = nn.Linear(n_feature, config.n_class)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'bn' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'bn' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'fc' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        out = self.features(x)
        out = self.activation(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
