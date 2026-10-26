"""Reference-based EEG architectures. Input: (batch, 1, electrodes, time).

Architecture sources and deliberate adaptations are documented in MODEL_AUDIT.md.
All models return logits. Call apply_constraints() after each optimizer update.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES = ('EEGNet', 'DeepConvNet', 'ShallowConvNet', 'SCCNet', 'MBEEGSE')


def activation(name):
    return {'elu': nn.ELU, 'relu': nn.ReLU,
            'leakyrelu': lambda: nn.LeakyReLU(0.01), 'gelu': nn.GELU,
            'square': Square}[name]()


class Square(nn.Module):
    def forward(self, x):
        return x.float().square()


class SafeLog(nn.Module):
    def forward(self, x):
        return x.float().clamp_min(1e-6).log()


class SamePad(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, x):
        return F.pad(x, ((self.width-1)//2, self.width//2, 0, 0))


def flatten_channels_last(x):
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)


def initialize(module):
    """Glorot initialization; depthwise fans use Keras kernel dimensions."""
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if isinstance(layer, nn.Conv2d) and layer.groups == layer.in_channels and layer.groups > 1:
                area = math.prod(layer.kernel_size)
                multiplier = layer.out_channels // layer.in_channels
                bound = math.sqrt(6 / (area * (layer.in_channels + multiplier)))
                nn.init.uniform_(layer.weight, -bound, bound)
            else:
                nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.constraints = []
        self.l2_layers = []

    @torch.no_grad()
    def apply_constraints(self):
        for layer, dims, limit in self.constraints:
            w = layer.weight
            norm = w.square().sum(dim=dims, keepdim=True).sqrt()
            w.mul_(norm.clamp(max=limit) / (norm + 1e-7))

    def regularization_loss(self):
        return sum(coef * layer.weight.square().sum() for layer, coef in self.l2_layers)

    def predict_proba(self, x):
        return self(x).softmax(dim=1)


class EEGFeatures(nn.Module):
    def __init__(self, channels, f1=8, d=2, kernel=64, dropout=0.5,
                 act='elu', pool1=4, pool2=8):
        super().__init__()
        f2 = f1*d
        self.spatial = nn.Conv2d(f1, f2, (channels, 1), groups=f1, bias=False)
        self.temporal = nn.Sequential(SamePad(kernel),
            nn.Conv2d(1, f1, (1, kernel), bias=False),
            nn.BatchNorm2d(f1, eps=1e-3, momentum=0.01))
        self.spatial_block = nn.Sequential(self.spatial,
            nn.BatchNorm2d(f2, eps=1e-3, momentum=0.01), activation(act),
            nn.AvgPool2d((1, pool1)), nn.Dropout(dropout))
        self.separable = nn.Sequential(SamePad(16),
            nn.Conv2d(f2, f2, (1, 16), groups=f2, bias=False),
            nn.Conv2d(f2, f2, 1, bias=False),
            nn.BatchNorm2d(f2, eps=1e-3, momentum=0.01), activation(act),
            nn.AvgPool2d((1, pool2)), nn.Dropout(dropout))

    def forward(self, x):
        return self.separable(self.spatial_block(self.temporal(x)))


class EEGNet(BaseModel):
    def __init__(self, channels=2, samples=750, classes=2, dropout=None, act='native', **_):
        super().__init__()
        act = 'elu' if act == 'native' else act
        self.features = EEGFeatures(channels, dropout=0.5 if dropout is None else dropout, act=act)
        self.classifier = nn.Linear(16*(samples//4//8), classes)
        self.constraints = [(self.features.spatial, (2,), 1.), (self.classifier, (1,), 0.25)]
        initialize(self)

    def forward(self, x):
        return self.classifier(flatten_channels_last(self.features(x)))


class DeepConvNet(BaseModel):
    def __init__(self, channels=2, samples=750, classes=2, dropout=None,
                 act='native', profile='paper', **_):
        super().__init__()
        kernel, pool = (10, 3) if profile == 'paper' else (5, 2)
        act = 'elu' if act == 'native' else act
        p = 0.5 if dropout is None else dropout
        self.blocks = nn.ModuleList()
        first = nn.Conv2d(1, 25, (1, kernel), bias=True)
        spatial = nn.Conv2d(25, 25, (channels, 1), bias=False)
        self.blocks.append(nn.Sequential(first, spatial,
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1), activation(act),
            nn.MaxPool2d((1, pool)), nn.Dropout(p)))
        self.constraints.extend([(first, (1, 2, 3), 2.), (spatial, (1, 2, 3), 2.)])
        length = (samples-kernel+1)//pool
        for cin, cout in ((25, 50), (50, 100), (100, 200)):
            conv = nn.Conv2d(cin, cout, (1, kernel), bias=False)
            self.blocks.append(nn.Sequential(conv,
                nn.BatchNorm2d(cout, eps=1e-5, momentum=0.1), activation(act),
                nn.MaxPool2d((1, pool)), nn.Dropout(p)))
            self.constraints.append((conv, (1, 2, 3), 2.))
            length = (length-kernel+1)//pool
        if length < 1:
            raise ValueError('Input is too short for DeepConvNet; use longer trials or reference125 profile')
        self.classifier = nn.Linear(200*length, classes)
        self.constraints.append((self.classifier, (1,), 0.5))
        initialize(self)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.classifier(x.flatten(1))


class ShallowConvNet(BaseModel):
    def __init__(self, channels=2, samples=750, classes=2, dropout=None,
                 act='native', profile='paper', **_):
        super().__init__()
        kernel, pool, stride = (25, 75, 15) if profile == 'paper' else (13, 35, 7)
        temporal = nn.Conv2d(1, 40, (1, kernel), bias=True)
        spatial = nn.Conv2d(40, 40, (channels, 1), bias=False)
        self.features = nn.Sequential(temporal, spatial,
            nn.BatchNorm2d(40, eps=1e-5, momentum=0.1),
            activation('square' if act == 'native' else act),
            nn.AvgPool2d((1, pool), stride=(1, stride)), SafeLog(),
            nn.Dropout(0.5 if dropout is None else dropout))
        length = (samples-kernel+1-pool)//stride+1
        if length < 1:
            raise ValueError('Input is too short for ShallowConvNet')
        self.classifier = nn.Linear(40*length, classes)
        self.constraints = [(temporal, (1, 2, 3), 2.),
                            (spatial, (1, 2, 3), 2.), (self.classifier, (1,), 0.5)]
        initialize(self)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class SCCNet(BaseModel):
    def __init__(self, channels=2, samples=750, classes=2, dropout=None,
                 act='native', **_):
        super().__init__()
        # Fixed paper setting Nu=22, Nt=1, 20 spatiotemporal filters at 125 Hz.
        spatial = nn.Conv2d(1, 22, (channels, 1), bias=True)
        temporal = nn.Conv2d(22, 20, (1, 12), bias=True)
        self.features = nn.Sequential(spatial,
            nn.BatchNorm2d(22, eps=1e-3, momentum=0.01), SamePad(12), temporal,
            nn.BatchNorm2d(20, eps=1e-3, momentum=0.01),
            activation('square' if act == 'native' else act),
            nn.Dropout(0.5 if dropout is None else dropout),
            nn.AvgPool2d((1, 62), stride=(1, 12)), SafeLog())
        length = (samples-62)//12+1
        self.classifier = nn.Linear(20*length, classes)
        self.l2_layers = [(spatial, 1e-4), (temporal, 1e-4)]
        initialize(self)

    def forward(self, x):
        return self.classifier(flatten_channels_last(self.features(x)))


class SEBlock(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels//ratio)
        self.fc2 = nn.Linear(channels//ratio, channels)

    def forward(self, x):
        scale = self.fc2(F.relu(self.fc1(x.mean(dim=(2, 3))))).sigmoid()
        return x * scale[:, :, None, None]


class MBEEGSE(BaseModel):
    def __init__(self, channels=2, samples=750, classes=2, dropout=None, act='native', **_):
        super().__init__()
        # The coauthor's reference implementation uses pooling 8 then 8, D=2.
        self.branches = nn.ModuleList()
        act = 'elu' if act == 'native' else act
        for f1, kernel, p, ratio in ((4, 16, 0., 4), (8, 32, 0.1, 4), (16, 64, 0.2, 2)):
            features = EEGFeatures(channels, f1=f1, d=2, kernel=kernel,
                dropout=p if dropout is None else dropout, act=act, pool1=8, pool2=8)
            self.branches.append(nn.Sequential(features, SEBlock(f1*2, ratio)))
            # The reference has time as its first kernel axis: max_norm(axis=0)
            # on a (1,C) spatial kernel constrains individual coefficients.
            self.constraints.append((features.spatial, (3,), 1.))
        self.classifier = nn.Linear((8+16+32)*(samples//8//8), classes)
        self.constraints.append((self.classifier, (1,), 0.25))
        initialize(self)
        for layer in self.modules():
            if isinstance(layer, SEBlock):
                for fc in (layer.fc1, layer.fc2):
                    # Keras he_normal uses a variance-corrected truncated normal.
                    std = math.sqrt(2 / fc.in_features) / 0.8796256610342398
                    nn.init.trunc_normal_(fc.weight, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        features = [flatten_channels_last(branch(x)) for branch in self.branches]
        return self.classifier(torch.cat(features, dim=1))


def build_model(name, channels=2, samples=750, classes=2, **kwargs):
    if name not in MODEL_NAMES:
        raise ValueError(f'Unknown model {name}')
    return globals()[name](channels=channels, samples=samples, classes=classes, **kwargs)
