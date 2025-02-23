import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torchvision.models as models
from torch_fidelity import calculate_metrics
import logging
from config import GAN_LATENT_DIM, GAN_PRETRAIN_LR, GAN_FINETUNE_LR, GAN_BETA1, GAN_PRETRAIN_EPOCHS, GAN_BATCH_SIZE, GAN_GP_LAMBDA, GAN_CRITIC_ITERS, LOOKBACK_PERIOD, GAN_FID_THRESHOLD, GAN_MSE_THRESHOLD, GAN_ACF_THRESHOLD, GAN_STABILITY_THRESHOLD, GAN_METRIC_UPDATE_FREQ

# Setup logging
logging.basicConfig(filename='logs/bot.log', level=logging.INFO)
metric_logger = logging.getLogger('gan_metrics')
metric_handler = logging.FileHandler('logs/gan_metrics.log')
metric_handler.setLevel(logging.INFO)
metric_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
metric_handler.setFormatter(metric_formatter)
metric_logger.addHandler(metric_handler)
metric_logger.setLevel(logging.INFO)

class GANModel:
    def __init__(self, input_shape=(LOOKBACK_PERIOD, 2), learning_rate=GAN_PRETRAIN_LR, beta1=GAN_BETA1, latent_dim=GAN_LATENT_DIM):
        """Initialize WGAN-GP with tunable hyperparameters, GPU support, and enhanced logging."""
        self.input_shape = input_shape
        self.latent_dim = int(latent_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing GANModel on device: {self.device}")
        self.generator = self._build_generator().to(self.device)
        self.critic = self._build_critic().to(self.device)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        self.lambda_gp = GAN_GP_LAMBDA
        self.critic_iters = GAN_CRITIC_ITERS
        self.batch_size = GAN_BATCH_SIZE
        self.is_pretrained = False
        self.critic_losses = []
        self.generator_losses = []
        self.metric_logger = metric_logger

    def _build_generator(self):
        """Build a deeper generator network for synthetic data generation using PyTorch."""
        class Generator(nn.Module):
            def __init__(self, latent_dim, output_shape):
                super(Generator, self).__init__()
                self.latent_dim = latent_dim
                self.output_shape = output_shape
                self.model = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(128),
                    nn.Linear(128, 256),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, 512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, output_shape[0] * output_shape[1]),
                    nn.LeakyReLU(0.2)
                )
                self.reshape = lambda x: x.view(-1, *output_shape)

            def forward(self, x):
                x = self.model(x)
                return self.reshape(x)

        return Generator(self.latent_dim, self.input_shape)

    def _build_critic(self):
        """Build a deeper critic network for WGAN-GP using PyTorch."""
        class Critic(nn.Module):
            def __init__(self, input_shape):
                super(Critic, self).__init__()
                self.input_shape = input_shape
                self.model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_shape[0] * input_shape[1], 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, 1)  # Linear output for Wasserstein
                )

            def forward(self, x):
                return self.model(x)

        return Critic(self.input_shape)

    def _gradient_penalty(self, real_samples, fake_samples):
        """Compute gradient penalty to enforce Lipschitz constraint in WGAN-GP."""
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        pred = self.critic(interpolated)
        grads = torch.autograd.grad(outputs=pred, inputs=interpolated,
                                    grad_outputs=torch.ones_like(pred, device=self.device),
                                    create_graph=True, retain_graph=True)[0]
        norm = torch.sqrt(torch.sum(grads ** 2, dim=[1, 2]) + 1e-10)
        gp = torch.mean((norm - 1.0) ** 2)
        return gp

    def _train_critic(self, real_samples, fake_samples):
        """Train the critic with Wasserstein loss and gradient penalty."""
        self.critic_optimizer.zero_grad()
        real_output = self.critic(real_samples)
        fake_output = self.critic(fake_samples)
        c_loss = torch.mean(fake_output) - torch.mean(real_output)
        gp = self._gradient_penalty(real_samples, fake_samples)
        total_loss = c_loss + self.lambda_gp * gp
        total_loss.backward()
        self.critic_optimizer.step()
        return total_loss.item()

    def _train_generator(self, noise):
        """Train the generator to minimize critic's output."""
        self.generator_optimizer.zero_grad()
        fake_samples = self.generator(noise)
        fake_output = self.critic(fake_samples)
        g_loss = -torch.mean(fake_output)
        g_loss.backward()
        self.generator_optimizer.step()
        return g_loss.item()

    def _compute_fid(self, real_samples, fake_samples):
        """Compute FID using torch-fidelity with Inception V3 for precise quality assessment."""
        real_np = real_samples.cpu().numpy() if real_samples.is_cuda else real_samples.numpy()
        fake_np = fake_samples.cpu().numpy() if fake_samples.is_cuda else fake_samples.numpy()
        real_tensor = torch.tensor(real_np).float().permute(0, 2, 1).unsqueeze(1)
        fake_tensor = torch.tensor(fake_np).float().permute(0, 2, 1).unsqueeze(1)
        real_tensor = nn.functional.interpolate(real_tensor, size=(299, 299), mode='bilinear', align_corners=False)
        fake_tensor = nn.functional.interpolate(fake_tensor, size=(299, 299), mode='bilinear', align_corners=False)
        real_tensor = (real_tensor - real_tensor.min()) / (real_tensor.max() - real_tensor.min() + 1e-10)
        fake_tensor = (fake_tensor - fake_tensor.min()) / (fake_tensor.max() - fake_tensor.min() + 1e-10)
        metrics = calculate_metrics(input1=real_tensor, input2=fake_tensor, cuda=torch.cuda.is_available(), fid=True)
        return metrics['frechet_inception_distance']

    def _compute_mse_moments(self, real_samples, fake_samples):
        """Compute MSE of statistical moments for data fidelity."""
        from scipy.stats import skew, kurtosis
        real_mean = np.mean(real_samples.cpu().numpy(), axis=(0, 1))
        fake_mean = np.mean(fake_samples.cpu().numpy(), axis=(0, 1))
        real_var = np.var(real_samples.cpu().numpy(), axis=(0, 1))
        fake_var = np.var(fake_samples.cpu().numpy(), axis=(0, 1))
        real_skew = np.mean([skew(sample.flatten()) for sample in real_samples.cpu().numpy()])
        fake_skew = np.mean([skew(sample.flatten()) for sample in fake_samples.cpu().numpy()])
        real_kurt = np.mean([kurtosis(sample.flatten()) for sample in real_samples.cpu().numpy()])
        fake_kurt = np.mean([kurtosis(sample.flatten()) for sample in fake_samples.cpu().numpy()])
        mse = (np.mean((real_mean - fake_mean) ** 2) + 
               np.mean((real_var - fake_var) ** 2) + 
               (real_skew - fake_skew) ** 2 + 
               (real_kurt - fake_kurt) ** 2)
        return mse

    def _compute_acf_error(self, real_samples, fake_samples):
        """Compute autocorrelation error across lags 1-5 for temporal realism."""
        def acf(data, lag):
            mean = np.mean(data)
            var = np.var(data)
            if var == 0:
                return 0
            shifted = data[lag:] - mean
            original = data[:-lag] - mean
            return np.mean(shifted * original) / var

        lags = range(1, min(6, LOOKBACK_PERIOD))
        real_acfs = [np.mean([acf(sample.flatten(), lag) for sample in real_samples.cpu().numpy()]) for lag in lags]
        fake_acfs = [np.mean([acf(sample.flatten(), lag) for sample in fake_samples.cpu().numpy()]) for lag in lags]
        acf_error = np.mean([abs(r - f) for r, f in zip(real_acfs, fake_acfs)])
        return acf_error

    def _compute_stability(self):
        """Compute stability as average variance of critic and generator losses."""
        c_stability = np.var(self.critic_losses[-100:]) if len(self.critic_losses) >= 100 else 0
        g_stability = np.var(self.generator_losses[-100:]) if len(self.generator_losses) >= 100 else 0
        return (c_stability + g_stability) / 2

    def train(self, real_data, epochs=GAN_PRETRAIN_EPOCHS, batch_size=None):
        """Train the WGAN-GP with enhanced metric logging on GPU if available."""
        real_data = np.array(real_data)
        batch_size = batch_size or self.batch_size
        num_batches = max(1, len(real_data) // batch_size)

        if self.is_pretrained:
            for param_group in self.generator_optimizer.param_groups:
                param_group['lr'] = GAN_FINETUNE_LR
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = GAN_FINETUNE_LR
            epochs = 5
            self.logger.info("Switching to GAN fine-tuning mode with reduced LR")

        for epoch in range(epochs):
            c_losses_epoch = []
            g_losses_epoch = []
            for _ in range(num_batches):
                for _ in range(self.critic_iters):
                    idx = np.random.randint(0, real_data.shape[0], batch_size)
                    real_samples = torch.tensor(real_data[idx], dtype=torch.float32, device=self.device)
                    noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                    fake_samples = self.generator(noise)
                    c_loss = self._train_critic(real_samples, fake_samples)
                    c_losses_epoch.append(c_loss)

                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                g_loss = self._train_generator(noise)
                g_losses_epoch.append(g_loss)

            self.critic_losses.extend(c_losses_epoch)
            self.generator_losses.extend(g_losses_epoch)

            if (epoch + 1) % GAN_METRIC_UPDATE_FREQ == 0 or epoch == epochs - 1:
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_samples = self.generator(noise)
                idx = np.random.randint(0, real_data.shape[0], batch_size)
                real_samples = torch.tensor(real_data[idx], dtype=torch.float32, device=self.device)

                fid = self._compute_fid(real_samples, fake_samples)
                mse_moments = self._compute_mse_moments(real_samples, fake_samples)
                acf_error = self._compute_acf_error(real_samples, fake_samples)
                stability = self._compute_stability()

                self.logger.info(f"GAN Epoch {epoch + 1}/{epochs} - C Loss: {c_loss:.4f}, G Loss: {g_loss:.4f}")
                self.metric_logger.info(f"Epoch {epoch + 1}/{epochs} - FID: {fid:.2f} "
                                       f"(Min: {min([fid] + self.critic_losses[-100:]):.2f}, "
                                       f"Max: {max([fid] + self.critic_losses[-100:]):.2f}, "
                                       f"Mean: {np.mean([fid] + self.critic_losses[-100:]):.2f})")
                self.metric_logger.info(f"Epoch {epoch + 1}/{epochs} - MSE Moments: {mse_moments:.4f} "
                                       f"(Min: {min([mse_moments] + self.critic_losses[-100:]):.4f}, "
                                       f"Max: {max([mse_moments] + self.critic_losses[-100:]):.4f}, "
                                       f"Mean: {np.mean([mse_moments] + self.critic_losses[-100:]):.4f})")
                self.metric_logger.info(f"Epoch {epoch + 1}/{epochs} - ACF Error: {acf_error:.4f} "
                                       f"(Min: {min([acf_error] + self.critic_losses[-100:]):.4f}, "
                                       f"Max: {max([acf_error] + self.critic_losses[-100:]):.4f}, "
                                       f"Mean: {np.mean([acf_error] + self.critic_losses[-100:]):.4f})")
                self.metric_logger.info(f"Epoch {epoch + 1}/{epochs} - Stability: {stability:.4f} "
                                       f"(Min: {min([stability] + self.critic_losses[-100:]):.4f}, "
                                       f"Max: {max([stability] + self.critic_losses[-100:]):.4f}, "
                                       f"Mean: {np.mean([stability] + self.critic_losses[-100:]):.4f})")

                if fid > GAN_FID_THRESHOLD:
                    self.metric_logger.warning(f"FID {fid:.2f} exceeds threshold {GAN_FID_THRESHOLD}")
                if mse_moments > GAN_MSE_THRESHOLD:
                    self.metric_logger.warning(f"MSE Moments {mse_moments:.4f} exceeds threshold {GAN_MSE_THRESHOLD}")
                if acf_error > GAN_ACF_THRESHOLD:
                    self.metric_logger.warning(f"ACF Error {acf_error:.4f} exceeds threshold {GAN_ACF_THRESHOLD}")
                if stability > GAN_STABILITY_THRESHOLD:
                    self.metric_logger.warning(f"Stability {stability:.4f} exceeds threshold {GAN_STABILITY_THRESHOLD}")

                yield {
                    'fid': fid,
                    'mse_moments': mse_moments,
                    'acf_error': acf_error,
                    'stability': stability,
                    'epoch': epoch + 1
                }

        if not self.is_pretrained:
            self.is_pretrained = True

    def generate_synthetic_data(self, num_samples=1):
        """Generate synthetic market data with tuned WGAN-GP on GPU if available."""
        noise = torch.randn(num_samples, self.latent_dim, device=self.device)
        with torch.no_grad():
            synthetic_data = self.generator(noise)
        return synthetic_data.cpu().numpy()  # Return as numpy array for compatibility