"""A hierarchical deep convolutional VAE model.
Based on https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_hierarchical_example.ipynb
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from hvae.models import VAE
from hvae.models.blocks import MLP
from hvae.utils.dct import reconstruct_dct


class HVAE(VAE):
    """Conditional hierarchical VAE"""

    def __init__(self, num_classes: int = None, num_levels: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.num_hidden = 32

        self.r_nets = nn.ModuleList(
            [
                MLP(
                    dims=[
                        self.encoder_output_size,
                        self.encoder_output_size,
                    ],
                )
                for _ in range(self.num_levels)
            ]
        )
        self.delta_nets = nn.ModuleList(
            [
                MLP(
                    dims=[
                        self.encoder_output_size,
                        2 * self.latent_dim,
                    ]
                )
                for _ in range(self.num_levels)
            ]
        )

        self.z_nets = nn.ModuleList(
            [
                MLP(
                    dims=[
                        self.latent_dim + self.num_classes,
                        2 * self.latent_dim,
                    ]
                )
                for _ in range(self.num_levels - 1)
            ]
            + [nn.Identity()]
        )

        self.decoder_input = MLP(
            dims=[
                self.latent_dim + num_classes,
                self.num_hidden,
                self.encoder_output_size,
            ]
        )
        self._h = nn.Parameter(torch.randn(self.latent_dim))

    @property
    def h(self):
        return self._h.to(self.device)

    def configure_optimizers(self):
        """Configure the optimizers."""
        params = [
            {"params": self.parameters(), "lr": self.lr},
        ]
        return torch.optim.Adam(params)

    def step(self, batch):
        x, y = batch
        outputs = self.forward(x, y)
        outputs["x"] = x
        loss = self.loss_function(**outputs)
        return loss, outputs["x_hat"]

    def forward(self, x: Tensor, y: Tensor, level: int = 0) -> list[Tensor]:
        """Perform the forward pass.
        Args:
            x: Input tensor of shape (B x C x H x W)
        Returns:
            A dictionary of tensors (x_hat, mu_log_vars, mu_log_var_deltas)
        """
        assert level < self.num_levels, f"Invalid level: {level}."
        y = F.one_hot(y, num_classes=self.num_classes).float().to(self.device)

        x = self.encoder(x).flatten(start_dim=1)
        rs = []
        for net in self.r_nets:
            x = net(x)
            rs.append(x)

        mu_log_var_deltas = []
        for r, net in zip(rs, self.delta_nets):
            delta_mu, delta_log_var = torch.chunk(net(r), 2, dim=1)
            delta_log_var = F.hardtanh(delta_log_var, -7.0, 2.0)
            mu_log_var_deltas.append((delta_mu, delta_log_var))

        zs = []
        mu_log_vars = []
        previous_z = None
        for (delta_mu, delta_log_var), net in zip(
            reversed(mu_log_var_deltas), reversed(self.z_nets)
        ):
            if previous_z is None:
                previous_z = torch.ones(self.latent_dim).to(self.device)
                mu = torch.zeros_like(delta_mu).to(self.device)
                log_var = torch.ones_like(delta_log_var).to(self.device)
            else:
                mu, log_var = torch.chunk(
                    net(torch.cat([previous_z, y], dim=1)), 2, dim=1
                )
            mu_log_vars.append((mu, log_var))
            z = self.reparameterize(mu + delta_mu, log_var + delta_log_var) + previous_z
            assert not torch.isnan(z).any(), "NaN in z."
            assert not torch.isnan(mu).any(), "NaN in mu."
            assert not torch.isnan(log_var).any(), "NaN in log_var."
            assert not torch.isnan(delta_mu).any(), "NaN in delta_mu."
            assert not torch.isnan(delta_log_var).any(), "NaN in delta_log_var."
            zs.append(z)
            previous_z = z
        zs = list(reversed(zs))
        mu_log_vars = list(reversed(mu_log_vars))

        x_hat = self.decode(self.before_decoder(zs, y, level=level))

        return {
            "x_hat": x_hat,
            "mu_log_vars": mu_log_vars,
            "mu_log_var_deltas": mu_log_var_deltas,
        }

    def loss_function(
        self,
        x: Tensor,
        x_hat: Tensor,
        mu_log_vars: list[Tensor],
        mu_log_var_deltas: list[Tensor],
    ) -> dict:
        """Compute the loss given ground truth images and their reconstructions.
        Args:
            x: Ground truth images of shape (B x C x H x W)
            x_hat: Reconstructed images of shape (B x C x H x W)
            mu: Latent mean of shape (B x D)
            log_var: Latent log variance of shape (B x D)
            kld_weight: Weight for the Kullback-Leibler divergence term
        Returns:
            Dictionary containing the loss value and the individual losses
        """
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]

        klds = []
        for (mu, log_var), (delta_mu, delta_log_var) in zip(
            mu_log_vars, mu_log_var_deltas
        ):
            if mu is not None:
                klds.append(
                    0.5 * delta_mu**2 / torch.exp(log_var)
                    + torch.exp(delta_log_var)
                    - delta_log_var
                    - 1
                )
            else:
                klds.append(
                    0.5 * delta_mu**2 + torch.exp(delta_log_var) - delta_log_var - 1
                )

        kld = sum(klds).sum() / x.shape[0]
        loss = reconstruction_loss + self.beta * kld

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_divergence": kld,
        }

    @torch.no_grad()
    def sample(
        self, num_samples: int, z: Tensor = None, y: Tensor = None, level: int = 0
    ) -> Tensor:
        """Sample a vector in the latent space and return the corresponding image.
        Args:
            num_samples: Number of samples to generate
            current_device: Device to run the model
        Returns:
            Tensor of shape (num_samples x C x H x W)
        """
        assert level < self.num_levels, f"Invalid level: {level}."
        if y is None:
            y = (
                torch.randint(self.num_classes, size=(num_samples,))
                .to(self.device)
                .long()
            )
        else:
            assert y.shape[0] == num_samples, "Invalid number of samples."
        y = F.one_hot(y, num_classes=self.num_classes).float().to(self.device)

        if z is None:
            z = torch.randn(num_samples, self.latent_dim).to(self.device)

        zs = [z]
        for net in reversed(self.z_nets[:-1]):
            z = net(torch.cat([z, y], dim=1))
            mu, log_var = torch.chunk(z, 2, dim=1)
            z = self.reparameterize(mu, log_var)
            zs.append(z)
        zs = list(reversed(zs))

        return self.decode(self.before_decoder(zs, y, level=level))

    def before_decoder(self, zs: list[Tensor], y: Tensor, level: int = 0):
        """Concatenate the latent vectors with a one-hot encoding of y."""
        z = torch.cat([zs[0], y], dim=1)
        return self.decoder_input(z)

    def generate_noise(self, num_samples: int) -> Tensor:
        """Generate a noise tensor to use for sampling."""
        return torch.randn(num_samples, self.latent_dim).to(self.device)


class DCTHVAE(HVAE):
    """Conditional hierarchical VAE with DCT reconstruction."""

    def __init__(self, ks: list[int], **kwargs):
        super().__init__(num_levels=len(ks), **kwargs)
        self.ks = ks
        self.decoder_input = MLP(
            dims=[
                self.latent_dim + self.num_classes,
                self.num_hidden,
                self.encoder_output_size,
            ]
        )

    def step(self, batch):
        x, y = batch
        losses = []
        for level, k in enumerate(self.ks):
            x_dct = reconstruct_dct(x, k=k).to(self.device)
            outputs = self.forward(x_dct, y, level=level)
            outputs["x"] = x_dct
            losses.append(self.loss_function(**outputs))
        loss = {k: sum(loss[k] for loss in losses) for k in losses[0].keys()}
        return loss, outputs["x_hat"]
