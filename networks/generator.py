from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import init_weights
from typing import Tuple


class GeneratorBase(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorBase, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.input_dim = input_dim
        self.output_dim = output_dim

    # @abstractmethod
    def forward_(self, batch_size: int, n_lags: int, device: str):
        """Implement here generation scheme."""
        pass

    def forward(self, batch_size: int, n_lags: int, device: str):
        x = self.forward_(batch_size, n_lags, device)
        x = self.pipeline.inverse_transform(x)
        return x

# Sample model based on LSTM
class GeneratorBase2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorBase2, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward_(self, batch_size: int, n_lags: int, device: str):
        """ Implement here generation scheme. """
        # ...
        pass

class LSTMGenerator(GeneratorBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        noise_scale=0.1,
        BM=False,
        activation=nn.Tanh(),
    ):
        super(LSTMGenerator, self).__init__(input_dim, output_dim)
        # LSTM
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.rnn.apply(init_weights)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)
        self.linear.apply(init_weights)

        self.initial_nn = nn.Sequential(
            ResFNN(input_dim, hidden_dim * n_layers, [hidden_dim, hidden_dim]),
            nn.Tanh(),
        )  # we use a simple residual network to learn the distribution at the initial time step.
        self.initial_nn1 = nn.Sequential(
            ResFNN(input_dim, hidden_dim * n_layers, [hidden_dim, hidden_dim]),
            nn.Tanh(),
        )
        self.initial_nn.apply(init_weights)
        self.initial_nn1.apply(init_weights)

        self.BM = BM
        if BM:
            self.noise_scale = noise_scale
        else:
            self.noise_scale    = 0.3
        self.activation = activation

    def forward(
        self, batch_size: int, n_lags: int, device: str, z=None
    ) -> torch.Tensor:
        if z == None:
            z = (self.noise_scale * torch.randn(batch_size, n_lags, self.input_dim)).to(
                device
            )  # cumsum(1)
            if self.BM:
                z = z.cumsum(1)
            else:
                pass
            # z[:, 0, :] *= 0  # first point is fixed
            #
        else:
            z = z
        z0 = self.noise_scale * torch.randn(batch_size, self.input_dim, device=device)

        h0 = (
            self.initial_nn(z0)
            .view(batch_size, self.rnn.num_layers, self.rnn.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        c0 = (
            self.initial_nn1(z0)
            .view(batch_size, self.rnn.num_layers, self.rnn.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        # c0 = torch.zeros_like(h0)

        h1, _ = self.rnn(z, (h0, c0))
        x = self.linear(self.activation(h1))

        assert x.shape[1] == n_lags

        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.LeakyReLU()
        self.create_residual_connection = True if input_dim == output_dim else False

    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.create_residual_connection:
            y = x + y
        return y


class ResFNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int],
        flatten: bool = False,
    ):
        """
        Feedforward neural network with residual connection.
        Args:
            input_dim: integer, specifies input dimension of the neural network
            output_dim: integer, specifies output dimension of the neural network
            hidden_dims: list of integers, specifies the hidden dimensions of each layer.
                in above definition L = len(hidden_dims) since the last hidden layer is followed by an output layer
        """
        super(ResFNN, self).__init__()
        blocks = list()
        self.input_dim = input_dim
        self.flatten = flatten
        input_dim_block = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(ResidualBlock(input_dim_block, hidden_dim))
            input_dim_block = hidden_dim
        blocks.append(torch.nn.Tanh())
        blocks.append(nn.Linear(input_dim_block, output_dim))
        self.network = nn.Sequential(*blocks)
        self.blocks = blocks

    def forward(self, x):
        if self.flatten:
            x = x.reshape(x.shape[0], -1)
        out = self.network(x)
        return out
    
class ConditionalLSTMGenerator(GeneratorBase2):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 n_layers: int, init_fixed: bool = True):
        super(ConditionalLSTMGenerator, self).__init__(input_dim, output_dim)
        # LSTM
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
        self.linear.apply(init_weights)
        # neural network to initialise h0 from the LSTM
        # we put a tanh at the end because we are initialising h0 from the LSTM, that needs to take values between [-1,1]

        self.init_fixed = init_fixed

    def forward(self, batch_size: int, n_lags: int, device: str,
                condition=None, z=None) -> torch.Tensor:
        if self.init_fixed:
            h0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(device)
        else:
            h0 = torch.randn(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(device).requires_grad_()

        if condition is not None:
            z = (0.1 * torch.randn(batch_size, n_lags, self.input_dim - condition.shape[-1])).to(device)
            z[:, 0, :] *= 0
            z = z.cumsum(1)
            z = torch.cat([z, condition.unsqueeze(1).repeat((1, n_lags, 1))], dim=2)
        else:
            if z is None:
                z = (0.1 * torch.randn(batch_size, n_lags, self.input_dim)).to(device)
            z[:, 0, :] *= 0
            z = z.cumsum(1)

        c0 = torch.zeros_like(h0)
        h1, _ = self.rnn(z, (h0, c0))
        x = self.linear(h1)

        assert x.shape[1] == n_lags
        return x


class ConditionalLSTMGeneratorConstraint(GeneratorBase2):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 n_layers: int, init_fixed: bool = True, vol_activation: str = 'softplus'):
        super(ConditionalLSTMGeneratorConstraint, self).__init__(input_dim, output_dim)
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
        self.linear.apply(init_weights)
        self.init_fixed = init_fixed

        # Activation function for volatility
        if vol_activation == 'relu':
            self.activation_fn = F.relu
        elif vol_activation == 'softplus':
            self.activation_fn = F.softplus
        else:
            raise ValueError("Unsupported activation. Choose 'relu' or 'softplus'.")

    def forward(self, batch_size: int, n_lags: int, device: str,
                condition=None, z=None) -> torch.Tensor:
        if self.init_fixed:
            h0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(device)
        else:
            h0 = torch.randn(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(device).requires_grad_()

        if condition is not None:
            z = (0.1 * torch.randn(batch_size, n_lags, self.input_dim - condition.shape[-1])).to(device)
            z[:, 0, :] *= 0
            z = z.cumsum(1)
            z = torch.cat([z, condition.unsqueeze(1).repeat((1, n_lags, 1))], dim=2)
        else:
            if z is None:
                z = (0.1 * torch.randn(batch_size, n_lags, self.input_dim)).to(device)
            z[:, 0, :] *= 0
            z = z.cumsum(1)

        c0 = torch.zeros_like(h0)
        h1, _ = self.rnn(z, (h0, c0))
        x = self.linear(h1)

        # Ensure returns and volatility constraints
        log_returns = x[..., ::2]
        volatility = self.activation_fn(x[..., 1::2])

        # Combine log returns and volatility back
        final_out = torch.zeros_like(x)
        final_out[..., ::2] = log_returns
        final_out[..., 1::2] = volatility
        assert final_out.shape[1] == n_lags

        return final_out