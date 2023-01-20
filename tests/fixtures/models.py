# pylint: disable=too-few-public-methods
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class IdentityModel(nn.Module):
    """Identity Model."""

    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x: Any) -> Any:
        return self.identity(x)


class SingleInputNet(nn.Module):
    """Simple CNN model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MultipleInputNetDifferentDtypes(nn.Module):
    """Model with multiple inputs containing different dtypes."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)

        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = x2.type(torch.float)
        x2 = F.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = torch.cat((x1, x2), 0)
        return F.log_softmax(x, dim=1)


class ScalarNet(nn.Module):
    """Model that takes a scalar as a parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.identity = IdentityModel()

    def forward(self, x: torch.Tensor, scalar: float) -> torch.Tensor:
        out = x
        scalar = self.identity(scalar)
        if scalar == 5:
            out = self.conv1(out)
        else:
            out = self.conv2(out)
        return out


class EdgeCaseModel(nn.Module):
    """Model that throws an exception when used."""

    def __init__(
        self,
        throw_error: bool = False,
        return_str: bool = False,
        return_class: bool = False,
    ) -> None:
        super().__init__()
        self.throw_error = throw_error
        self.return_str = return_str
        self.return_class = return_class
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.model = IdentityModel()

    def forward(self, x: torch.Tensor) -> Any:
        x = self.conv1(x)
        x = self.model("string output" if self.return_str else x)
        if self.throw_error:
            x = self.conv1(x)
        if self.return_class:
            x = self.model(EdgeCaseModel)
        return x


class MLP(nn.Module):
    """Multi Layer Perceptron with inplace option.
    Make sure inplace=true and false has the same visual graph"""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace),
            nn.Linear(64, 32),
            nn.ReLU(inplace),
            nn.Linear(32, 16),
            nn.ReLU(inplace),
            nn.Linear(16, 8),
            nn.ReLU(inplace),
            nn.Linear(8, 4),
            nn.ReLU(inplace),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x


class LSTMNet(nn.Module):
    """Batch-first LSTM model."""

    def __init__(
        self,
        vocab_size: int = 20,
        embed_dim: int = 300,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # We use batch_first=False here.
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers)  # noqa: E501
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embedding(x)
        out, hidden = self.encoder(embed)
        out = self.decoder(out)
        out = out.view(-1, out.size(2))
        return out, hidden


class RecursiveNet(nn.Module):
    """Model that uses a layer recursively in computation."""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.activation = nn.ReLU(inplace)

    def forward(
        self, x: torch.Tensor, args1: Any = None, args2: Any = None
    ) -> torch.Tensor:
        del args1, args2
        out = x
        for _ in range(3):
            out = self.activation(self.conv1(out))
            out = self.conv1(out)
        return out


class SimpleRNN(nn.Module):
    """Simple RNN"""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.hid_dim = 2
        self.input_dim = 3
        self.max_length = 4
        self.lstm = nn.LSTMCell(self.input_dim, self.hid_dim)
        self.activation = nn.LeakyReLU(inplace=inplace)
        self.projection = nn.Linear(self.hid_dim, self.input_dim)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        b_size = token_embedding.size()[0]
        hx = torch.randn(b_size, self.hid_dim, device=token_embedding.device)
        cx = torch.randn(b_size, self.hid_dim, device=token_embedding.device)

        for _ in range(self.max_length):
            hx, cx = self.lstm(token_embedding, (hx, cx))
            hx = self.activation(hx)

        return hx


class SiameseNets(nn.Module):
    """Model with MaxPool and ReLU layers."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)

        self.pooling = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.pooling(F.relu(self.conv1(x1)))
        x1 = self.pooling(F.relu(self.conv2(x1)))
        x1 = self.pooling(F.relu(self.conv3(x1)))
        x1 = self.pooling(F.relu(self.conv4(x1)))

        x2 = self.pooling(F.relu(self.conv1(x2)))
        x2 = self.pooling(F.relu(self.conv2(x2)))
        x2 = self.pooling(F.relu(self.conv3(x2)))
        x2 = self.pooling(F.relu(self.conv4(x2)))

        batch_size = x1.size(0)
        x1 = x1.view(batch_size, -1)
        x2 = x2.view(batch_size, -1)
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)

        metric = torch.abs(x1 - x2)
        similarity = torch.sigmoid(self.fc2(self.dropout(metric)))
        return similarity


class FunctionalNet(nn.Module):
    """Model that uses many functional torch layers."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1600, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 1600)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class RecursiveRelu(nn.Module):
    '''Model with many recursive layers'''

    def __init__(self, seq_len: int = 8) -> None:
        super().__init__()

        self.activation = nn.ReLU(inplace=True)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for _ in range(self.seq_len):
            x = self.activation(x)

        return x


class Tower(nn.Module):
    '''Tower Model'''
    def __init__(self, length: int = 1) -> None:
        super().__init__()
        self.layers = []
        for i in range(length):
            lazy_layer = nn.LazyLinear(out_features=10)
            self.add_module(f"tower{i}", lazy_layer)
            self.layers.append(lazy_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l_layer in self.layers:
            x = l_layer(x)
        return x


class TowerBranches(nn.Module):
    '''Model with different length of tower used for expand_nested'''
    def __init__(self) -> None:
        super().__init__()
        self.tower1 = Tower(2)
        self.tower2 = Tower(3)
        self.tower3 = Tower(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.add(self.tower1(x) + self.tower2(x), self.tower3(x))


class OutputReused(nn.Module):
    """Multi Layer Perceptron with inplace option.
    Make sure inplace=true and false has the same visual graph"""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace),
            nn.Linear(64, 32),
            nn.ReLU(inplace),
            nn.Linear(32, 16),
            nn.ReLU(inplace),
            nn.Linear(16, 8),
            nn.ReLU(inplace),
            nn.Linear(8, 4),
            nn.ReLU(inplace),
            nn.Linear(4, 2),
        )

        self.empty = nn.Identity(())
        self.act = nn.ReLU(inplace)

    def forward(
        self, x1: torch.Tensor,
        x2: torch.Tensor, x3: torch.Tensor,
        x4: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.layers(x1)
        x = x + x2
        y = self.empty(self.act(x2)) + x3
        # x3 += 1
        return x, y, x3, x4


class InputNotUsed(nn.Module):
    """Multi Layer Perceptron with inplace option.
    Make sure inplace=true and false has the same visual graph"""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace),
            nn.Linear(64, 32),
            nn.ReLU(inplace),
            nn.Linear(32, 16),
            nn.ReLU(inplace),
            nn.Linear(16, 8),
            nn.ReLU(inplace),
            nn.Linear(8, 4),
            nn.ReLU(inplace),
            nn.Linear(4, 2),
        )

        self.empty = nn.Identity(())
        self.act = nn.ReLU(inplace)

    def forward(
        self, x1: torch.Tensor,
        x2: torch.Tensor, x3: torch.Tensor,
        x4: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor,]:
        x = self.layers(x1)
        x = x + x2
        y = self.empty(self.act(x2)) + x3
        return x, y


class CreateTensorsInside(nn.Module):
    '''Module that creates tensor during forward prop'''
    def __init__(self,) -> None:
        super().__init__()
        self.layer1 = nn.Linear(10, 30)
        self.layer2 = nn.Linear(30, 50)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x += torch.abs(torch.ones(1, 1))
        x = self.layer2(x)

        return x


class EnsembleMLP(nn.Module):
    """Multi Layer Perceptron with inplace option.
    Make sure inplace=true and false has the same visual graph"""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.num_model = 16
        self.mlp_layers = [
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(inplace),
                nn.Linear(64, 32),
                nn.ReLU(inplace),
                nn.Linear(32, 16),
            )
            for _ in range(self.num_model)
        ]

        self.fc_layer = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> Any:
        x_arr = [
            mlp(x) for mlp in self.mlp_layers
        ]
        x_concated = torch.cat(x_arr, dim=0)

        out = self.fc_layer(x_concated)
        return out


class Detect(nn.Module):

    def __init__(self, nc: int, ch: tuple[int, ...]) -> None:
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16
        self.no = sum(ch) * 2

        self.cv2 = nn.ModuleList(
            nn.Conv2d(x, self.reg_max * 4, 3, padding='same') for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 3, padding='same') for x in ch)

    def forward(
        self, x: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | list[torch.Tensor]:
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((x[i]+1, x[i]+2), 1)
        if self.training:
            return x
        return torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2), x


class DetectionWrapper(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.detector = Detect(nc=80, ch=(192, 192))

    def forward(
        self, *x: tuple[torch.Tensor]
    ) -> Any:
        y = list(x)
        return self.detector(y)


class IsolatedTensor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> Any:
        y = torch.zeros(1, 1)
        y_value = y.item()
        out = self.lin(x)
        out = out + y_value
        return out
