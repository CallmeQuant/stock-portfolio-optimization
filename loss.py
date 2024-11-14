import torch
import numpy as np
from torch import nn
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
import copy
from dataclasses import dataclass
from utils import to_numpy


def gaussian_kernel(X, Y, sigma):
    """
    Compute the Gaussian (RBF) kernel between X and Y.
    """
    pairwise_distances = torch.cdist(X, Y, p=2)
    K = torch.exp(-pairwise_distances ** 2 / (2 * sigma ** 2))
    return K

def mmd_loss(X: torch.Tensor, Y: torch.Tensor, sigma=1.0) -> torch.Tensor:
    '''
    Compute the unbiased MMD loss between X and Y using a Gaussian kernel.
    
    Parameters:
    X: torch.Tensor of shape (n_samples, n_features)
    Y: torch.Tensor of shape (n_samples, n_features)
    sigma: float, bandwidth parameter for the Gaussian kernel
    
    Returns:
    torch.Tensor scalar representing the MMD loss.
    '''
    # Calculate Gram matrices
    K_XX = gaussian_kernel(X, X, sigma)
    K_YY = gaussian_kernel(Y, Y, sigma)
    K_XY = gaussian_kernel(X, Y, sigma)
    
    # Unbiased MMD statistic
    n = K_XX.shape[0]
    m = K_YY.shape[0]
    
    # Zero the diagonal entries
    K_XX = K_XX - torch.diag(torch.diag(K_XX))
    K_YY = K_YY - torch.diag(torch.diag(K_YY))
    
    mmd = (torch.sum(K_XX) / (n * (n - 1))
           + torch.sum(K_YY) / (m * (m - 1))
           - 2 * torch.sum(K_XY) / (n * m))
    
    return mmd


def psd_loss(real_data, generated_data):
    """
    Compute the difference in power spectral densities between real and generated data.
    
    Parameters:
    real_data: torch.Tensor of shape (batch_size, sequence_length, features)
    generated_data: torch.Tensor of shape (batch_size, sequence_length, features)
    
    Returns:
    torch.Tensor scalar representing the loss.
    """
    # Compute the Fourier transforms
    real_fft = torch.fft.fft(real_data, dim=1)
    gen_fft = torch.fft.fft(generated_data, dim=1)
    
    # Compute the power spectral densities
    real_psd = torch.abs(real_fft) ** 2
    gen_psd = torch.abs(gen_fft) ** 2
    
    # Average over the batch and features
    real_psd_mean = real_psd.mean(dim=0)
    gen_psd_mean = gen_psd.mean(dim=0)
    
    # Compute the difference
    psd_diff = real_psd_mean - gen_psd_mean
    
    # Compute loss as Frobenius norm of the difference
    loss = torch.norm(psd_diff, p='fro')
    
    return loss

def calculate_acf_difference(real_data, generated_data, max_lag=5):
    """
    Compute the difference in autocorrelation functions between real and generated data.
    
    Parameters:
    real_data: torch.Tensor of shape (batch_size, sequence_length, features)
    generated_data: torch.Tensor of shape (batch_size, sequence_length, features)
    max_lag: int, maximum lag to compute the ACF
    
    Returns:
    torch.Tensor scalar representing the loss.
    """
    batch_size, sequence_length, num_features = real_data.shape
    
    # Initialize lists to hold ACFs for each feature
    real_acfs = []
    gen_acfs = []
    
    for feature_idx in range(num_features):
        # Extract the time series for the current feature
        real_series = real_data[:, :, feature_idx]  # Shape: (batch_size, sequence_length)
        gen_series = generated_data[:, :, feature_idx]
        
        # Compute mean over batch and time
        real_mean = real_series.mean()
        gen_mean = gen_series.mean()
        
        # Center the series
        real_centered = real_series - real_mean
        gen_centered = gen_series - gen_mean
        
        # Compute variance
        real_var = real_centered.var(unbiased=False) + 1e-8
        gen_var = gen_centered.var(unbiased=False) + 1e-8
        
        # Compute autocorrelation for each lag
        real_acf = []
        gen_acf = []
        for lag in range(1, max_lag + 1):
            # Ensure we have enough data points
            if lag >= sequence_length:
                break
            
            # Compute autocovariance
            real_cov = (real_centered[:, :-lag] * real_centered[:, lag:]).mean()
            gen_cov = (gen_centered[:, :-lag] * gen_centered[:, lag:]).mean()
            
            # Normalize to get autocorrelation
            real_acf.append(real_cov / real_var)
            gen_acf.append(gen_cov / gen_var)
        
        # Stack ACFs for the current feature
        real_acfs.append(torch.stack(real_acf))
        gen_acfs.append(torch.stack(gen_acf))
    
    # Stack ACFs across all features
    real_acfs = torch.stack(real_acfs)  # Shape: (num_features, effective_max_lag)
    gen_acfs = torch.stack(gen_acfs)
    
    # Compute difference
    acf_diff = real_acfs - gen_acfs
    
    # Compute loss as Frobenius norm of the difference
    loss = torch.norm(acf_diff, p='fro')
    
    return loss
  
def calculate_correlation_difference(real_data, generated_data):
    """
    Compute the difference in correlation matrices between real and generated data.
    
    Parameters:
    real_data: torch.Tensor of shape (batch_size, n_features)
    generated_data: torch.Tensor of shape (batch_size, n_features)
    
    Returns:
    torch.Tensor scalar representing the loss.
    """
    # Compute covariance matrices
    real_cov = torch.cov(real_data.T)
    gen_cov = torch.cov(generated_data.T)
    
    # Compute standard deviations
    real_std = torch.sqrt(torch.diag(real_cov) + 1e-8)
    gen_std = torch.sqrt(torch.diag(gen_cov) + 1e-8)
    
    # Compute correlation matrices
    real_corr = real_cov / torch.ger(real_std, real_std)
    gen_corr = gen_cov / torch.ger(gen_std, gen_std)
    
    # Compute difference
    corr_diff = real_corr - gen_corr
    
    # Compute loss as Frobenius norm of the difference
    loss = torch.norm(corr_diff, p='fro')
    
    return loss

def acf_diff(x): return torch.sqrt(torch.pow(x, 2).sum(0))
def cc_diff(x): return torch.abs(x).sum(0)
def cov_diff(x): return torch.abs(x).mean()

class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x, seed=None):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo
        self.seed = seed

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)

class ACFLoss(Loss):
    def __init__(self, x_real, max_lag=64, stationary=True, **kwargs):
        super(ACFLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.stationary = stationary
        self.metric = AutoCorrelationMetric(self.transform)
        self.acf_calc = lambda x: self.metric.measure(x, self.max_lag, stationary,dim=(0, 1),symmetric=False)
        self.acf_real = self.acf_calc(x_real)

    def compute(self, x_fake):
        acf_fake = self.acf_calc(x_fake)
        return self.norm_foo(acf_fake - self.acf_real.to(x_fake.device))


class MeanLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(MeanLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.mean = x_real.mean((0, 1))

    def compute(self, x_fake, **kwargs):
        return self.norm_foo(x_fake.mean((0, 1)) - self.mean)


class StdLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(StdLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.std_real = x_real.std((0, 1))

    def compute(self, x_fake, **kwargs):
        return self.norm_foo(x_fake.std((0, 1)) - self.std_real)


class CrossCorrelLoss(Loss):
    def __init__(self, x_real, max_lag=64, **kwargs):
        super(CrossCorrelLoss, self).__init__(norm_foo=cc_diff, **kwargs)
        self.lags = max_lag
        self.metric = CrossCorrelationMetric(self.transform)
        self.cross_correl_real = self.metric.measure(x_real,self.lags).mean(0)[0]
        self.max_lag = max_lag

    def compute(self, x_fake):
        cross_correl_fake = self.metric.measure(x_fake,lags=self.lags).mean(0)[0]
        loss = self.norm_foo(
            cross_correl_fake - self.cross_correl_real.to(x_fake.device)).unsqueeze(0)
        return loss


# unused
class cross_correlation(Loss):
    def __init__(self, x_real, **kwargs):
        super(cross_correlation).__init__(**kwargs)
        self.x_real = x_real

    def compute(self, x_fake):
        fake_corre = torch.from_numpy(np.corrcoef(
            x_fake.mean(1).permute(1, 0))).float()
        real_corre = torch.from_numpy(np.corrcoef(
            self.x_real.mean(1).permute(1, 0))).float()
        return torch.abs(fake_corre-real_corre)


def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    b = b+1e-5 if b == a else b
    # delta = (b - a) / n_bins
    bins = torch.linspace(a, b, n_bins+1)
    delta = bins[1]-bins[0]
    # bins = torch.arange(a, b + 1.5e-5, step=delta)
    count = torch.histc(x, bins=n_bins, min=a, max=b).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


class HistoLoss(Loss):

    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            tmp_densities = list()
            tmp_locs = list()
            tmp_deltas = list()
            # Exclude the initial point
            for t in range(x_real.shape[1]):
                x_ti = x_real[:, t, i].reshape(-1, 1)
                d, b = histogram_torch(x_ti, n_bins, density=True)
                tmp_densities.append(nn.Parameter(d).to(x_real.device))
                delta = b[1:2] - b[:1]
                loc = 0.5 * (b[1:] + b[:-1])
                tmp_locs.append(loc)
                tmp_deltas.append(delta)
            self.densities.append(tmp_densities)
            self.locs.append(tmp_locs)
            self.deltas.append(tmp_deltas)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            tmp_loss = list()
            # Exclude the initial point
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1).to(x_fake.device)
                x_ti = x_fake[:, t, i].contiguous(
                ).view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                counter = (relu(self.deltas[i][t].to(
                    x_fake.device) / 2. - dist) > 0.).float()
                density = counter.mean(0) / self.deltas[i][t].to(x_fake.device)
                abs_metric = torch.abs(
                    density - self.densities[i][t].to(x_fake.device))
                loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise


class CovLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(CovLoss, self).__init__(norm_foo=cov_diff, **kwargs)
        self.metric = CovarianceMetric(self.transform)
        self.covariance_real = self.metric.measure(x_real)
    def compute(self, x_fake):
        covariance_fake = self.metric.measure(x_fake)
        loss = self.norm_foo(covariance_fake -
                             self.covariance_real.to(x_fake.device))
        return loss


class VARLoss(Loss):
    def __init__(self, x_real, alpha=0.05, **kwargs):
        name = kwargs.pop('name')
        super(VARLoss, self).__init__(name=name)
        self.alpha = alpha
        self.var = tail_metric(x=x_real, alpha=self.alpha, statistic='var')

    def compute(self, x_fake):
        loss = list()
        var_fake = tail_metric(x=x_fake, alpha=self.alpha, statistic='var')
        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                abs_metric = torch.abs(var_fake[i][t] - self.var[i][t].to(x_fake.device))
                loss.append(abs_metric)
        loss_componentwise = torch.stack(loss)
        return loss_componentwise

class ESLoss(Loss):
    def __init__(self, x_real, alpha=0.05, **kwargs):
        name = kwargs.pop('name')
        super(ESLoss, self).__init__(name=name)
        self.alpha = alpha
        self.var = tail_metric(x=x_real, alpha=self.alpha, statistic='es')

    def compute(self, x_fake):
        loss = list()
        var_fake = tail_metric(x=x_fake, alpha=self.alpha, statistic='es')
        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                abs_metric = torch.abs(var_fake[i][t] - self.var[i][t].to(x_fake.device))
                loss.append(abs_metric)
        loss_componentwise = torch.stack(loss)
        return loss_componentwise

def tail_metric(x, alpha, statistic):
    res = list()
    for i in range(x.shape[2]):
        tmp_res = list()
        # Exclude the initial point
        for t in range(x.shape[1]):
            x_ti = x[:, t, i].reshape(-1, 1)
            sorted_arr, _ = torch.sort(x_ti)
            var_alpha_index = int(alpha * len(sorted_arr))
            var_alpha = sorted_arr[var_alpha_index]
            if statistic == "es":
                es_values = sorted_arr[:var_alpha_index + 1]
                es_alpha = es_values.mean()
                tmp_res.append(es_alpha)
            else:
                tmp_res.append(var_alpha)
        res.append(tmp_res)
    return res

from typing import Tuple, Optional
import torch
from abc import ABC, abstractmethod

'''
Define metrics classes for loss and score computation
Metric List:
- CovarianceMetric
- AutoCorrelationMetric
- CrossCorrelationMetric
- HistogramMetric
- SignatureMetric: SigW1Metric, SigMMDMetric

'''

class Metric(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    def measure(self,data, **kwargs):
        pass


class CovarianceMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'CovMetric'

    def measure(self,data):
        return cov_torch(self.transform(data))

class AutoCorrelationMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'AcfMetric'

    def measure(self,data,max_lag,stationary,dim=(0, 1),symmetric=False):
        if stationary:
            return acf_torch(self.transform(data),max_lag=max_lag,dim=dim)
        else:
            return non_stationary_acf_torch(self.transform(data),symmetric).to(data.device)


class CrossCorrelationMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'CrossCorrMetric'

    def measure(self,data,lags,dim=(0, 1)):
        return cacf_torch(self.transform(data),lags,dim)


class MeanAbsDiffMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'MeanAbsDiffMetric'

    def measure(self,data):
        x1, x2 = self.transform(data)
        return mean_abs_diff(x1,x2)


class MMDMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'MMDMetric'

    def measure(self,data):
        x1, x2 = self.transform(data)
        return mmd(x1,x2)


class ONNDMetric(Metric):

    def __init__(self,transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'ONNDMetric'

    def measure(self,data: Tuple[torch.Tensor,torch.Tensor]):
        """
        Calculates the Outgoing Nearest Neighbour Distance (ONND) to assess the diversity of the generated data
        Parameters
        ----------
        x_real: torch.tensor, [B, L, D]
        x_fake: torch.tensor, [B, L', D']

        Returns
        -------
        ONND: float
        """
        x_real, x_fake = data
        b1, t1, d1 = x_real.shape
        b2, t2, d2 = x_fake.shape
        assert t1 == t2, "Time length does not agree!"
        assert d1 == d2, "Feature dimension does not agree!"

        # Compute samplewise difference
        x_real_repeated = x_real.repeat_interleave(b2, 0)
        x_fake_repeated = x_fake.repeat([b1, 1, 1])
        samplewise_diff = x_real_repeated - x_fake_repeated
        # Compute samplewise MSE
        MSE_X_Y = torch.norm(samplewise_diff, dim=2).mean(dim=1).reshape([b1, -1])
        # For every sample in x_real, compute the minimum MSE and calculate the average among all the minimums
        ONND = (torch.min(MSE_X_Y, dim=1)[0]).mean()
        return ONND


class INNDMetric(Metric):

    def __init__(self, transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'INNDMetric'

    def measure(self, data: Tuple[torch.Tensor, torch.Tensor]):
        """
        Calculates the Incoming Nearest Neighbour Distance (INND) to assess the authenticity of the generated data
        Parameters
        ----------
        x_real: torch.tensor, [B, L, D]
        x_fake: torch.tensor, [B, L', D']

        Returns
        -------
        INND: float
        """
        x_real, x_fake = data
        b1, t1, d1 = x_real.shape
        b2, t2, d2 = x_fake.shape
        assert t1 == t2, "Time length does not agree!"
        assert d1 == d2, "Feature dimension does not agree!"

        # Compute samplewise difference
        x_fake_repeated = x_fake.repeat_interleave(b1, 0)
        x_real_repeated = x_real.repeat([b2, 1, 1])
        samplewise_diff = x_real_repeated - x_fake_repeated
        # Compute samplewise MSE
        MSE_X_Y = torch.norm(samplewise_diff, dim=2).mean(dim=1).reshape([b2, -1])
        # For every sample in x_real, compute the minimum MSE and calculate the average among all the minimums
        INND = (torch.min(MSE_X_Y, dim=0)[0]).mean()
        return INND


class ICDMetric(Metric):

    def __init__(self, transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'INNDMetric'

    def measure(self, data: torch.Tensor):
        """
        Calculates the Intra Class Distance (ICD) to detect a potential model collapse
        Parameters
        ----------
        x_fake: torch.tensor, [B, L, D]

        Returns
        -------
        ICD: float
        """
        x_fake = data
        batch, _, _ = x_fake.shape

        # Compute samplewise difference
        x_fake_repeated_interleave = x_fake.repeat_interleave(batch, 0)
        x_fake_repeated = x_fake.repeat([batch, 1, 1])
        samplewise_diff = x_fake_repeated_interleave - x_fake_repeated
        # Compute samplewise MSE
        MSE_X_Y = torch.norm(samplewise_diff, dim=2).mean(dim=1).reshape([batch, -1])
        # For every sample in x_real, compute the minimum MSE and calculate the average among all the minimums
        ICD = 2 * (MSE_X_Y).sum()
        return ICD / (batch ** 2)


class VARMetric(Metric):
    def __init__(self, alpha=0.05, transform=lambda x: x):
        self.transform = transform
        self.alpha = alpha

    @property
    def name(self):
        return 'VARMetric'

    def measure(self, data: Tuple[torch.Tensor, torch.Tensor]):
        """
        Calculates the alpha-value at risk to assess the tail distribution match of the generated data
        Parameters
        ----------
        x_real: torch.tensor, [B, L, D]
        x_fake: torch.tensor, [B, L', D']

        Returns
        -------
        INND: float
        """
        x_fake = data
        batch, _, _ = x_fake.shape

        # Compute samplewise difference
        x_fake_repeated_interleave = x_fake.repeat_interleave(batch, 0)
        x_fake_repeated = x_fake.repeat([batch, 1, 1])
        samplewise_diff = x_fake_repeated_interleave - x_fake_repeated
        # Compute samplewise MSE
        MSE_X_Y = torch.norm(samplewise_diff, dim=2).mean(dim=1).reshape([batch, -1])
        # For every sample in x_real, compute the minimum MSE and calculate the average among all the minimums
        ICD = 2 * (MSE_X_Y).sum()
        return ICD / (batch ** 2)
    
def cov_torch(x):
    """Estimates covariance matrix like numpy.cov"""
    device = x.device
    x = to_numpy(x)
    _, L, C = x.shape
    x = x.reshape(-1, L * C)
    return torch.from_numpy(np.cov(x, rowvar=False)).to(device).float()


def acf_torch(x: torch.Tensor, max_lag: int, dim: Tuple[int] = (0, 1)) -> torch.Tensor:
    """
    :param x: torch.Tensor [B, S, D]
    :param max_lag: int. specifies number of lags to compute the acf for
    :return: acf of x. [max_lag, D]
    """
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def non_stationary_acf_torch(X, symmetric=False):
    """
    Compute the correlation matrix between any two time points of the time series
    Parameters
    ----------
    X (torch.Tensor): [B, T, D]
    symmetric (bool): whether to return the upper triangular matrix of the full matrix

    Returns
    -------
    Correlation matrix of the shape [T, T, D] where each entry (t_i, t_j, d_i) is the correlation between the d_i-th coordinate of X_{t_i} and X_{t_j}
    """
    # Get the batch size, sequence length, and input dimension from the input tensor
    B, T, D = X.shape

    # Create a tensor to hold the correlations
    correlations = torch.zeros(T, T, D)

    for i in range(D):
        # Compute the correlation between X_{t, d} and X_{t-tau, d}
        if hasattr(torch, 'corrcoef'):  # version >= torch2.0
            correlations[:, :, i] = torch.corrcoef(X[:, :, i].t())
        else:  # TODO: test and fix
            correlations[:, :, i] = torch.from_numpy(np.corrcoef(to_numpy(X[:, :, i]).T))

    if not symmetric:
        # Loop through each time step from lag to T-1
        for t in range(T):
            # Loop through each lag from 1 to lag
            for tau in range(t + 1, T):
                correlations[tau, t, :] = 0

    return correlations


def cacf_torch(x, lags: list, dim=(0, 1)):
    """
    Computes the cross-correlation between feature dimension and time dimension
    Parameters
    ----------
    x
    lags
    dim

    Returns
    -------

    """

    # Define a helper function to get the lower triangular indices for a given dimension
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    # Get the lower triangular indices for the input tensor x
    ind = get_lower_triangular_indices(x.shape[2])

    # Standardize the input tensor x along the given dimensions
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)

    # Split the input tensor into left and right parts based on the lower triangular indices
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]

    # Compute the cross-correlation at each lag and store in a list
    cacf_list = list()
    for i in range(lags):
        # Compute the element-wise product of the left and right parts, shifted by the lag if i > 0
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r

        # Compute the mean of the product along the time dimension
        cacf_i = torch.mean(y, (1))

        # Append the result to the list of cross-correlations
        cacf_list.append(cacf_i)

    # Concatenate the cross-correlations across lags and reshape to the desired output shape
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def rmse(x, y):
    return (x - y).pow(2).sum().sqrt()


def mean_abs_diff(den1: torch.Tensor, den2: torch.Tensor):
    return torch.mean(torch.abs(den1 - den2), 0)


def mmd(x, y):
    pass
