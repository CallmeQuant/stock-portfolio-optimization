import torch

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