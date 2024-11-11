import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Union

from baselines.base import GeneratorBase
from loss import *
from Generative.spectral_filter import SpectralFilter, DFT

class FourierFlow(GeneratorBase):
    def __init__(self, input_dim, output_dim, 
                 hidden, n_flows, n_lags, 
                 FFT=True, flip=True, normalize=False):
        super().__init__(input_dim, output_dim)

        self.FFT = FFT
        self.normalize = normalize
        self.n_flows = n_flows
        self.hidden = hidden

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.individual_shape = (n_lags, output_dim)
        self.d = np.prod(self.individual_shape)
        self.k = int(np.ceil(self.d / 2))

        # Configure flips for each flow
        self.flips = [True if i % 2 else False for i in range(n_flows)] if flip else [False for i in range(n_flows)]

        self.bijectors = None
        self.FourierTransform = None
        self.fft_mean = None
        self.fft_std = None
        self.carry_flag = False

    def forward(self, batch_size: int, n_lags: int):
        """
        Generate samples using the trained model.

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        n_lags : int
            Length of the sequence to generate.
        device : str (not used)
            Device to use for computation ('cpu' or 'cuda').

        Returns
        -------
        torch.Tensor
            Generated samples.
        """
        return self.sample(batch_size)


    def forward_step(self, x):
        """
        Perform one step of the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x : torch.Tensor
            Transformed tensor.
        log_pz : torch.Tensor
            Log probability of the transformed tensor.
        log_jacob : float
            Sum of log determinants of the Jacobian matrices.
        """
        if self.FFT:
            x = self.FourierTransform(x)[0]

            if self.normalize:
                x = (x - self.fft_mean) / (self.fft_std + 1e-8)

            x = x.view(-1, self.d)

        log_jacobs = []

        for bijector, f in zip(self.bijectors, self.flips):
            x, log_pz, lj = bijector(x, flip=f)
            log_jacobs.append(lj)

        return x, log_pz, sum(log_jacobs)

    def inverse(self, z):
        """
        Perform the inverse transformation.

        Parameters
        ----------
        z : torch.Tensor
            Transformed tensor.

        Returns
        -------
        numpy.ndarray
            Reconstructed input tensor.
        """
        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):
            z = bijector.inverse(z, flip=f)

        if self.FFT:
            if self.normalize:
                z = z * self.fft_std.view(-1, self.d) + self.fft_mean.view(-1, self.d)

            z = self.FourierTransform.inverse(z)

        return z.detach().numpy()

    def fit(self, X, epochs=500, batch_size=128, 
            learning_rate=1e-3, display_step=100):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        epochs : int, optional
            Number of training epochs (default is 500).
        batch_size : int, optional
            Batch size (default is 128).
        learning_rate : float, optional
            Learning rate (default is 1e-3).
        display_step : int, optional
            Interval to display training progress (default is 100).

        Returns
        -------
        list
            Training losses.
        """
        X_train = torch.from_numpy(np.array(X)).float()

        self.carry_flag = False
        if np.prod(X_train.shape[1:]) % 2 == 1:
            repeat_last = X_train[:, :, -1:]
            X_train = torch.cat([X_train, repeat_last], dim=2)
            self.carry_flag = True

        self.individual_shape = X_train.shape[1:]
        self.d = np.prod(self.individual_shape)
        self.k = int(np.ceil(self.d / 2))

        assert self.d % 2 == 0

        self.bijectors = nn.ModuleList(
            [
                SpectralFilter(self.d, self.k, self.FFT, hidden=self.hidden, flip=self.flips[_])
                for _ in range(self.n_flows)
            ]
        )

        self.FourierTransform = DFT(N_fft=self.d)
        X_train = X_train.reshape(-1, self.d)

        X_train_spectral = self.FourierTransform(X_train)[0]
        assert X_train_spectral.shape[-1] == self.k

        self.fft_mean = torch.mean(X_train_spectral, dim=0)
        self.fft_std = torch.std(X_train_spectral, dim=0)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)

        losses = []

        for step in tqdm(range(epochs), desc="Training Epochs"):
            optimizer.zero_grad()

            z, log_pz, log_jacob = self.forward_step(X_train)
            base_loss = (-log_pz - log_jacob).mean()

            generated_data = self.inverse(z)
            generated_data = torch.tensor(generated_data).reshape(-1, *self.individual_shape)

            if self.carry_flag:
                generated_data = generated_data[:, :, :-1]

            real_data = X_train.reshape(-1, *self.individual_shape)

            # Calculate additional losses
            correlation_loss = calculate_correlation_difference(
                real_data.view(real_data.shape[0], -1),
                generated_data.view(generated_data.shape[0], -1)
            )
            acf_loss = calculate_acf_difference(
                real_data,
                generated_data,
                max_lag=5
            )
            mmd_loss_value = mmd_loss(
                real_data.view(real_data.shape[0], -1),
                generated_data.view(generated_data.shape[0], -1),
                sigma=1.0 
            )
            psd_loss_value = psd_loss(
                real_data,
                generated_data
            )

            # Total loss
            total_loss = (base_loss
                          + correlation_loss
                          + acf_loss
                          + mmd_loss_value
                          + psd_loss_value)
            losses.append(total_loss.detach().numpy())

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if (step % display_step == 0) or (step == epochs - 1):
                print(f"{step}/{epochs} "
                      f"| Base Loss: {base_loss.item():.3f} "
                      f"| Correlation Loss: {correlation_loss.item():.3f} "
                      f"| ACF Loss: {acf_loss.item():.3f} "
                      f"| MMD Loss: {mmd_loss_value.item():.3f} "
                      f"| PSD Loss: {psd_loss_value.item():.3f} "
                      f"| Total Loss: {total_loss.item():.3f}")
            if step == epochs - 1:
                print(f"Final Total Loss: {total_loss.item():.3f}\n")
                print("Finished Training")

        return losses

    def sample(self, n_samples):
        """
        Sample new data points from the trained model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        device : str, optional
            Device to use for computation (default is 'cpu').

        Returns
        -------
        torch.Tensor
            Generated samples.
        """
        # Generate samples from the base distribution
        mu, cov = torch.zeros(self.d), torch.eye(self.d)
        p_Z = MultivariateNormal(mu, cov)
        z = p_Z.rsample(sample_shape=(n_samples,))

        # Transform samples back to data space
        X_sample = self.inverse(z)
        X_sample = torch.from_numpy(X_sample).float().reshape(-1, *self.individual_shape)

        # Remove padding if it was added
        if self.carry_flag:
            X_sample = X_sample[:, :, :-1]

        return X_sample