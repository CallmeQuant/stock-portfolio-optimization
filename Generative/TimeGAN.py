import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
import copy
from os import path as pt
from tqdm import tqdm

from utils import *
from baselines.base import BaseTrainer
from networks.discriminator import LSTMDiscriminator
from networks.generator import ConditionalLSTMGenerator
from loss import *

class TimeGAN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, out_dim,
                 activation=None):
        super(TimeGAN, self).__init__()
        self.input_dim = input_dim

        self.model = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                             num_layers=n_layers, batch_first=True, bidirectional=False)
        #self.linear = nn.Linear(hidden_dim*2, out_dim, bias=False)
        self.model.apply(init_weights)
        self.linear = nn.Linear(hidden_dim, out_dim)

        self.linear.apply(init_weights)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.model(x)[0]
        x = self.linear(h)
        if self.activation == None:
            return x
        else:
            return self.activation(x)

class TIMEGANTrainer(BaseTrainer):
    def __init__(self, G, gamma, train_dl, config, **kwargs):
        super(TIMEGANTrainer, self).__init__(
            G =G, G_optimizer=torch.optim.Adam(G.parameters(),
                                               lr = config.lr_G,
                                               betas=(0, 0.9)),
                                               **kwargs)

        self.config = config
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.D = TimeGAN(
            input_dim=config.input_dim, hidden_dim=config.D_hidden_dim,
            out_dim=1, n_layers=config.D_num_layers).to(config.device)
        self.embedder = TimeGAN(
            input_dim=config.input_dim, hidden_dim=config.D_hidden_dim,
            out_dim=config.input_dim, n_layers=config.D_num_layers,
            activation=nn.Sigmoid()).to(config.device)
        self.recovery = TimeGAN(
            input_dim=config.input_dim, hidden_dim=config.D_hidden_dim,
            out_dim=config.input_dim, n_layers=config.D_num_layers).to(config.device)
        self.supervisor = TimeGAN(
            input_dim=config.input_dim, hidden_dim=config.D_hidden_dim,
            out_dim=config.input_dim, n_layers=config.D_num_layers,
            activation=nn.Sigmoid()).to(config.device)
        self.D_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=config.lr_D, betas=(0, 0.9))
        self.embedder_optimizer = torch.optim.Adam(
            self.embedder.parameters(), lr=config.lr_D, betas=(0, 0.9))
        self.recovery_optimizer = torch.optim.Adam(
            self.recovery.parameters(), lr=config.lr_D, betas=(0, 0.9))
        self.supervisor_optimizer = torch.optim.Adam(
            self.supervisor.parameters(), lr=config.lr_D, betas=(0, 0.9))  # Using TTUR

        # Set up the loss function
        self.setup_losses(train_dl)

        self.gamma = gamma
        self.train_dl = train_dl
        self.reg_param = 0
        self.losses_history

    def setup_losses(self, train_dl):
        """Initialize all the different loss functions"""
        # Get a batch of real data to initialize losses
        X_real = next(iter(train_dl))[0].to(self.config.device)
        
        # Initialize different losses with configurable weights
        self.loss_functions = {
            'acf': ACFLoss(X_real, name='acf', max_lag=self.config.max_lag, 
                          reg=self.config.acf_weight),
            'mean': MeanLoss(X_real, name='mean', reg=self.config.mean_weight),
            'std': StdLoss(X_real, name='std', reg=self.config.std_weight),
            'corr': CrossCorrelLoss(X_real, name='corr', max_lag=self.config.max_lag, 
                                  reg=self.config.corr_weight),
            'cov': CovLoss(X_real, name='cov', reg=self.config.cov_weight),
            'hist': HistoLoss(X_real, name='hist', n_bins=self.config.hist_bins, 
                            reg=self.config.hist_weight),
            'var': VARLoss(X_real, name='var', alpha=self.config.var_alpha, 
                          reg=self.config.var_weight),
            'es': ESLoss(X_real, name='es', alpha=self.config.es_alpha, 
                        reg=self.config.es_weight)
        }

    def compute_statistical_loss(self, X_fake, X_real):
        """Compute combined statistical loss from all components"""
        total_loss = 0
        loss_components = {}
        
        for loss_name, loss_fn in self.loss_functions.items():
            if getattr(self.config, f'use_{loss_name}_loss', False):  # Check if this loss is enabled
                loss_value = loss_fn(X_fake)
                total_loss += loss_value
                loss_components[loss_name] = loss_value.item()
                
        return total_loss, loss_components

    def fit(self, device):
        self.G.to(device)
        self.D.to(device)
        self.supervisor.to(device)
        self.embedder.to(device)
        self.recovery.to(device)
        self.train_Embedder(device)
        self.train_supervisor(device)
        self.joint_train(device)

    # Train Embedding Mappings
    def train_Embedder(self, device):
        self.toggle_grad(self.embedder, True)
        self.toggle_grad(self.recovery, True)
        for i in tqdm(range(self.n_gradient_steps)):

            X = next(iter(self.train_dl))[0].to(device)
            X.requires_grad_()
            H = self.embedder(X)
            X_tilde = self.recovery(H)
            E_loss_T0 = nn.MSELoss()(X, X_tilde)
            E_loss0 = 10*torch.sqrt(E_loss_T0)
            self.embedder_optimizer.zero_grad()
            self.recovery_optimizer.zero_grad()
            E_loss0.backward(retain_graph=True)
            self.embedder_optimizer.step()
            self.recovery_optimizer.step()
        self.toggle_grad(self.embedder, False)
        self.toggle_grad(self.recovery, False)

    # Train Supervised Settings
    def train_supervisor(self, device):

        self.toggle_grad(self.G, True)
        self.toggle_grad(self.supervisor, True)
        self.G.train()
        self.supervisor.train()
        for i in tqdm(range(self.n_gradient_steps)):

            X = next(iter(self.train_dl))[0].to(device)
            X.requires_grad_()
            H = self.embedder(X)
            E_hat = self.G(batch_size=self.batch_size,
                           n_lags=self.config.n_lags, condition=None, device=device)
            H_hat_supervise = self.supervisor(H)
            G_loss_S = nn.MSELoss()(
                H[:, 1:, :], H_hat_supervise[:, :-1, :])
            self.G_optimizer.zero_grad()
            self.supervisor_optimizer.zero_grad()
            G_loss_S.backward(retain_graph=True)
            self.G_optimizer.step()
            self.supervisor_optimizer.step()
        self.toggle_grad(self.G, False)
        self.toggle_grad(self.supervisor, False)

    def joint_train(self, device):
        self.G.train()
        self.D.train()
        self.supervisor.train()
        self.embedder.train()
        self.recovery.train()
        for i in tqdm(range(self.n_gradient_steps)):
            for kk in range(2):
                self.toggle_grad(self.G, True)
                self.toggle_grad(self.supervisor, True)
                # generator
                X = next(iter(self.train_dl))[0].to(device)
                X.requires_grad_()
                H = self.embedder(X)
                X_tilde = self.recovery(H)
                E_hat = self.G(batch_size=self.batch_size,
                               n_lags=self.config.n_lags, condition=None, device=device)
                H_hat = self.supervisor(E_hat)
                H_hat_supervise = self.supervisor(H)

                X_hat = self.recovery(H_hat)

                # Discriminator
                Y_fake = self.D(H_hat)
                Y_fake_e = self.D(E_hat)

                # Generator loss
                # 1. Adversarial loss
                G_loss_U = self.compute_loss(Y_fake, 1.)
                G_loss_U_e = self.compute_loss(Y_fake_e, 1.)

                # 2. Supervised loss
                G_loss_S = nn.MSELoss()(
                    H[:, 1:, :], H_hat_supervise[:, :-1, :])

                # 3. Two Momments
                G_loss_V1 = torch.mean(torch.abs(
                    (torch.std(X_hat, [0], unbiased=False)) +\
                    1e-6 - (torch.std(X, [0]) + 1e-6)))
                G_loss_V2 = torch.mean(
                    torch.abs((torch.mean(X_hat, [0]) - (torch.mean(X, [0])))))

                G_loss_V = G_loss_V1 + G_loss_V2

                # 4. Statistical loss
                stat_loss, stat_components = self.compute_statistical_loss(X_hat, X)

                # 5. Summation
                G_loss = G_loss_U + self.gamma * G_loss_U_e + \
                    100 * torch.sqrt(G_loss_S) + 100*G_loss_V + stat_loss

                self.G_optimizer.zero_grad()
                self.supervisor_optimizer.zero_grad()

                G_loss.backward()
                self.G_optimizer.step()
                self.supervisor_optimizer.step()
                self.toggle_grad(self.G, False)
                self.toggle_grad(self.supervisor, False)
                self.toggle_grad(self.embedder, True)
                self.toggle_grad(self.recovery, True)
                H = self.embedder(X)
                X_tilde = self.recovery(H)
                H_hat_supervise = self.supervisor(H)
                G_loss_S = nn.MSELoss()(
                    H[:, 1:, :], H_hat_supervise[:, :-1, :])

                # Embedder network loss
                E_loss_T0 = nn.MSELoss()(X, X_tilde)
                E_loss0 = 10*torch.sqrt(E_loss_T0)
                E_loss = E_loss0 + 0.1*G_loss_S

                self.embedder_optimizer.zero_grad()
                self.recovery_optimizer.zero_grad()

                E_loss.backward()
                self.embedder_optimizer.step()
                self.recovery_optimizer.step()
                self.toggle_grad(self.embedder, False)
                self.toggle_grad(self.recovery, False)
            # discriminator
            self.toggle_grad(self.D, True)
            X = next(iter(self.train_dl))[0].to(device)
            E_hat = self.G(batch_size=self.batch_size,
                           n_lags=self.config.n_lags, condition=None, device=device)
            H_hat = self.supervisor(E_hat)
            H = self.embedder(X)

            # Discriminator
            Y_fake = self.D(H_hat)
            Y_real = self.D(H)
            Y_fake_e = self.D(E_hat)

            # On real data and fake data
            D_loss_real = self.compute_loss(Y_real, 1.)
            D_loss_fake = self.compute_loss(Y_fake, 0.)
            D_loss_fake_e = self.compute_loss(Y_fake_e, 0.)
            D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e
            self.D_optimizer.zero_grad()
            D_loss.backward(retain_graph=True)
            # Step discriminator params
            self.D_optimizer.step()
            self.toggle_grad(self.D, False)
            # self.evaluate(X_hat, X, i, self.config)
    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = torch.nn.BCELoss()(torch.nn.Sigmoid()(d_out), targets)
        return loss

    def save_model_dict(self):
        save_obj(self.G.state_dict(), pt.join(
            self.config.exp_dir, 'generator_state_dict.pt'))
        save_obj(self.embedder.state_dict(), pt.join(
            self.config.exp_dir, 'embedder_state_dict.pt'))
        save_obj(self.recovery.state_dict(), pt.join(
            self.config.exp_dir, 'recovery_state_dict.pt'))
        save_obj(self.supervisor.state_dict(), pt.join(
            self.config.exp_dir, 'supervisor_state_dict.pt'))

        if self.config.include_D:
            save_obj(self.D.state_dict(), pt.join(
                self.config.exp_dir, 'discriminator_state_dict.pt'))
            
def kfold_cross_validation(training_data, config, n_splits=2):
    kfold = KFold(n_splits=n_splits, shuffle=True)

    best_model_state = None
    best_loss = float('inf')

    for fold, (train_idx, val_idx) in enumerate(kfold.split(training_data)):
        print(f'Fold {fold + 1}/{n_splits}')

        # Prepare train and validation datasets
        cv_training_data = training_data[train_idx].to(config.device).to(torch.float)
        cv_validation_data = training_data[val_idx].to(config.device).to(torch.float)

        # Create datasets with just the data (no labels)
        train_set = TensorDataset(cv_training_data)
        val_set = TensorDataset(cv_validation_data)

        train_dl = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        val_dl = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
        
        generator = ConditionalLSTMGenerator(
            input_dim=config.G_input_dim, hidden_dim=config.G_hidden_dim,
            output_dim=config.input_dim,
            n_layers=config.G_num_layers
        )
        discriminator = LSTMDiscriminator(
            input_dim=config.input_dim, hidden_dim=config.D_hidden_dim,
            out_dim=1, n_layers=config.D_num_layers
        )

        # Initialize the trainer
        trainer = TIMEGANTrainer(G=generator, gamma=1,
                                train_dl=train_dl, batch_size=config.batch_size,
                                n_gradient_steps=config.steps,
                                config=config)

        # Train the model
        trainer.fit(config.device)

        # Evaluate on validation set
        val_loss = evaluate(trainer, val_dl, config.device)

        # Save the best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = {
                'G': copy.deepcopy(generator.state_dict()),
            }
            print(f'New best model found with validation loss: {val_loss}')

    # Save the best model to disk
    save_best_model(best_model_state, config)

def evaluate(trainer, val_dl, device):
    trainer.G.eval()
    trainer.embedder.eval()
    trainer.recovery.eval()
    trainer.supervisor.eval()
    trainer.D.eval() if trainer.config.include_D else None

    total_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for X, in val_dl: 
            X = X.to(device)
            H = trainer.embedder(X)
            X_tilde = trainer.recovery(H)
            loss = criterion(X, X_tilde)
            total_loss += loss.item() * X.size(0)

    avg_loss = total_loss / len(val_dl.dataset)
    return avg_loss

def save_best_model(model_state, config):
    save_obj(model_state['G'], './model_dict.pkl')