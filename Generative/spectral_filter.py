import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
from typing import List, Tuple, Dict

class SpectralFilter(nn.Module):
    """
    Spectral Filter torch module.

    Parameters
    ------------
    d : int
        Number of input dimensions.
    k : int
        Dimension of split in the input space.
    FFT : int
        Number of FFT components.
    hidden : int
        Number of hidden units in the spectral filter layer.
    flip : bool, optional
        Indicator whether to flip the split dimensions. Default is False.
    RNN : bool, optional
        Indicator whether to use an RNN in spectral filtering. Default is False.

    Attributes
    ----------
    out_size : int
        Output size after split.
    pz_size : int
        Size of the latent variable z.
    in_size : int
        Input size after split.
    sig_net : nn.Sequential
        Network for scaling.
    mu_net : nn.Sequential
        Network for translation.
    base_dist : torch.distributions.MultivariateNormal
        Base distribution for the flow.
    """
    def __init__(self, d: int, k: int, FFT: int, hidden: int, flip: bool=False, RNN: bool=False):
        super().__init__()
        self.d, self.k = d, k
        self.out_size = self.d - self.k
        self.pz_size = self.d
        self.in_size = self.k

        if flip:
            self.in_size, self.out_size = self.out_size, self.in_size

        self.sig_net = nn.Sequential(
            nn.Linear(self.in_size, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, self.out_size),
        )

        self.mu_net = nn.Sequential(
            nn.Linear(self.in_size, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, self.out_size),
        )

        base_mu, base_cov = torch.zeros(self.pz_size), torch.eye(self.pz_size)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def forward(self, x: torch.Tensor, flip: bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SpectralFilter module.

        Similar to RealNVP, see:
        Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio.
        "Density estimation using real NVP." arXiv preprint arXiv:1605.08803 (2016).

        Parameters
        ------------
        x : torch.Tensor
            Input tensor.
        flip : bool, optional
            Indicator whether to flip the split dimensions. Default is False.

        Returns
        ---------
        z_hat : torch.Tensor
            Transformed tensor after applying the spectral filter.
        log_pz : torch.Tensor
            Log probability of the transformed tensor under the base distribution.
        log_jacob : torch.Tensor
            Log determinant of the Jacobian of the transformation.
        """
        x1, x2 = x[:, : self.k], x[:, self.k :]
        if flip:
            x2, x1 = x1, x2

        # forward
        sig = self.sig_net(x1).view(-1, self.out_size)
        z1, z2 = x1, x2 * torch.exp(sig) + self.mu_net(x1).view(-1, self.out_size)

        if flip:
            z2, z1 = z1, z2

        z_hat = torch.cat([z1, z2], dim=-1)
        log_pz = self.base_dist.log_prob(z_hat)
        log_jacob = sig.sum(-1)

        return z_hat, log_pz, log_jacob

    def inverse(self, Z: torch.Tensor, flip: bool=False) -> torch.Tensor:
        """
        Inverse pass of the SpectralFilter module.

        Parameters
        ------------
        Z : torch.Tensor
            Input tensor in latent space.
        flip : bool, optional
            Indicator whether to flip the split dimensions. Default is False.

        Returns
        ---------
        x : torch.Tensor
            Reconstructed input tensor.
        """
        z1, z2 = Z[:, : self.k], Z[:, self.k :]
        if flip:
            z2, z1 = z1, z2

        x1 = z1
        sig_in = self.sig_net(z1).view(-1, self.out_size)
        x2 = (z2 - self.mu_net(z1).view(-1, self.out_size)) * torch.exp(-sig_in)

        if flip:
            x2, x1 = x1, x2

        return torch.cat([x1, x2], -1)


def flip(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Flipping helper function.

    Takes a vector as an input, then flips its elements from left to right along the specified dimension.

    Parameters
    ------------
    x : torch.Tensor
        Input tensor of size N x 1.
    dim : int
        Dimension along which to flip the elements.

    Returns
    ---------
    x_flipped : torch.Tensor
        Tensor with elements flipped along the specified dimension.
    """
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :,
        getattr(
            torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda]
        )().long(),
        :,
    ]
    return x.view(xsize)


def reconstruct_DFT(x: torch.Tensor, component: str="real") -> torch.Tensor:
    """
    Prepares input for the inverse DFT.

    Takes a cropped frequency component and creates a symmetric or anti-symmetric mirror of it before applying inverse DFT.

    Parameters
    ------------
    x : torch.Tensor
        Input tensor containing frequency components.
    component : str, optional
        Specifies whether the component is 'real' or 'imag'. Default is 'real'.

    Returns
    ---------
    x_rec : torch.Tensor
        Reconstructed tensor ready for inverse DFT.
    """
    if component == "real":
        x_rec = torch.cat([x[0, :], flip(x[0, :], dim=0)], dim=0)
    elif component == "imag":
        x_rec = torch.cat([x[1, :], -1 * flip(x[1, :], dim=0)], dim=0)
    return x_rec


# Main DFT module
class DFT(nn.Module):
    """
    Discrete Fourier Transform (DFT) torch module.

    Parameters
    ------------
    N_fft : int, optional
        Size of the DFT transform. Default is 100.

    Attributes
    ----------
    N_fft : int
        Size of the DFT transform.
    crop_size : int
        Size of non-redundant frequency components.
    base_dist : torch.distributions.MultivariateNormal
        Base distribution of the flow.
    """
    def __init__(self, N_fft: int=100):
        super(DFT, self).__init__()
        self.N_fft = N_fft
        self.crop_size = int(np.ceil(self.N_fft / 2))
        base_mu, base_cov = torch.zeros(self.crop_size * 2), torch.eye(
            self.crop_size * 2
        )
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Forward pass of the DFT module.

        Parameters
        ------------
        x : torch.Tensor
            Input tensor.

        Returns
        ---------
        x_fft : torch.Tensor
            Tensor containing the non-redundant frequency components after DFT.
        log_pz : torch.Tensor
            Log probability of the transformed tensor under the base distribution.
        log_jacob : int
            Log determinant of the Jacobian of the transformation (always zero for DFT).
        """
        if len(x.shape) == 1:
            x = x.reshape((1, -1))

        x_numpy = x.detach().float()
        X_fft = [np.fft.fftshift(np.fft.fft(x_numpy[k, :])) for k in range(x.shape[0])]
        X_fft_train = np.array(
            [
                np.array(
                    [
                        np.real(X_fft[k])[: self.crop_size] / self.N_fft,
                        np.imag(X_fft[k])[: self.crop_size] / self.N_fft,
                    ]
                )
                for k in range(len(X_fft))
            ]
        )
        x_fft = torch.from_numpy(X_fft_train).float()
        log_pz = self.base_dist.log_prob(
            x_fft.view(-1, x_fft.shape[1] * x_fft.shape[2])
        )
        log_jacob = 0
        return x_fft, log_pz, log_jacob

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass of the DFT module.

        Parameters
        ------------
        x : torch.Tensor
            Input tensor in frequency domain.

        Returns
        ---------
        x_ifft_out : torch.Tensor
            Reconstructed tensor in time domain.
        """
        x_numpy = x.view((-1, 2, self.crop_size))
        x_numpy_r = [
            reconstruct_DFT(x_numpy[u, :, :], component="real").detach().numpy()
            for u in range(x_numpy.shape[0])
        ]
        x_numpy_i = [
            reconstruct_DFT(x_numpy[u, :, :], component="imag").detach().numpy()
            for u in range(x_numpy.shape[0])
        ]

        x_ifft = [
            self.N_fft
            * np.real(np.fft.ifft(np.fft.ifftshift(x_numpy_r[u] + 1j * x_numpy_i[u])))
            for u in range(x_numpy.shape[0])
        ]
        x_ifft_out = torch.from_numpy(np.array(x_ifft)).float()
        return x_ifft_out