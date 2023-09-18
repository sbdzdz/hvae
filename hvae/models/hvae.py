"""A hierarchical deep convolutional VAE model.
Based on https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_hierarchical_example.ipynb
"""

import torch
from torch import Tensor
from torch.nn import functional as F

from hvae.models import VAE
from hvae.models.blocks import MLP
from hvae.utils.dct import reconstruct_dct


class HVAE(VAE):
    """Conditional hierarchical VAE"""

    def __init__(self, num_classes: int = None, num_levels: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_levels = num_levels

        self.r_nets = [
            MLP(
                dims=[
                    self.encoder_output_size,
                    self.encoder_output_size,
                ]
            ).to(self.device)
            for _ in range(self.num_levels)
        ]
        self.delta_nets = [
            MLP(
                dims=[
                    self.encoder_output_size,
                    2 * self.latent_dim,
                ]
            )
            for _ in range(self.num_levels)
        ]

        self.z_nets = [
            MLP(
                dims=[
                    self.latent_dim,
                    2 * self.latent_dim,
                ]
            )
            for _ in range(self.num_levels - 1)
        ] + [None]

        self.decoder_input = MLP(
            dims=[
                (self.num_levels * self.latent_dim) + num_classes,
                2 * self.num_levels * self.latent_dim,
                self.encoder_output_size,
            ]
        )

    def configure_optimizers(self):
        """Configure the optimizers."""
        params = [
            {"params": self.parameters(), "lr": self.lr},
            #    {"params": self.encoder.parameters(), "lr": self.lr},
            #    {"params": self.decoder_input.parameters(), "lr": self.lr},
            #    {"params": self.decoder.parameters(), "lr": self.lr},
        ]
        # params.extend(
        #    {"params": net.parameters(), "lr": self.lr}
        #    for net in self.r_nets + self.delta_nets + self.z_nets
        # )
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

        x = self.encoder(x).flatten(start_dim=1)
        rs = []
        for net in self.r_nets:
            x = net(x)
            rs.append(x.clone())

        mu_log_var_deltas = []
        for r, net in zip(rs, self.delta_nets):
            delta = net(r)
            delta_mu, delta_log_var = torch.chunk(delta, 2, dim=1)
            delta_log_var = F.hardtanh(delta_log_var, -7.0, 2.0)
            mu_log_var_deltas.append((delta_mu, delta_log_var))

        zs = []
        mu_log_vars = []
        previous_z = None
        for (delta_mu, delta_log_var), net in zip(
            reversed(mu_log_var_deltas), reversed(self.z_nets)
        ):
            if previous_z is None:
                mu_log_vars.append((None, None))
                z = self.reparameterize(delta_mu, delta_log_var)
            else:
                mu, log_var = torch.chunk(net(previous_z), 2, dim=1)
                mu_log_vars.append((mu, log_var))
                z = self.reparameterize(mu + delta_mu, log_var + delta_log_var)
            zs.append(z)
            previous_z = z
        zs = list(reversed(zs))
        mu_log_vars = list(reversed(mu_log_vars))
        for i in range(level):
            zs[i] = torch.zeros_like(zs[i])

        # concatenate all zs and a one-hot encoding of y
        y = F.one_hot(y, num_classes=self.num_classes).float().to(self.device)
        z = torch.cat([*zs, y], dim=1)

        # decode into the image space
        x_hat = self.decode(self.decoder_input(z))

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

        kl_divergences = []
        for mu, log_var, delta_mu, delta_log_var in zip(mu_log_vars, mu_log_var_deltas):
            if mu is not None:
                kl_divergences.append(
                    delta_mu**2 / torch.exp(log_var)
                    + torch.exp(delta_log_var)
                    - delta_log_var
                    - 1
                ).mean()
            else:
                kl_divergences.append(
                    delta_mu**2 + torch.exp(delta_log_var) - delta_log_var - 1
                ).mean()

        kl_divergence = sum(kl_divergences) / len(kl_divergences)
        loss = reconstruction_loss + self.beta * kl_divergence

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_divergence": kl_divergence,
        }

    @torch.no_grad()
    def sample(self, num_samples: int, y: Tensor = None, level: int = 0) -> Tensor:
        """Sample a vector in the latent space and return the corresponding image.
        Args:
            num_samples: Number of samples to generate
            current_device: Device to run the model
        Returns:
            Tensor of shape (num_samples x C x H x W)
        """
        assert level < self.num_levels, f"Invalid level: {level}."
        if y is None:
            y = torch.randint(self.num_classes, size=(num_samples,)).to(self.device)
        else:
            assert y.shape[0] == num_samples, "Invalid number of samples."

        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        zs = []
        for net in reversed(self.z_nets[:-1]):
            z = net(z)
            mu, log_var = torch.chunk(z, 2, dim=1)
            z = self.reparameterize(mu, log_var)
            zs.append(z.clone())
        zs = list(reversed(zs))

        for i in range(level):
            zs[i] = torch.zeros_like(zs[i])

        y = F.one_hot(y, num_classes=self.num_classes).float().to(self.device)
        z = torch.cat([*zs, y], dim=1)
        z = self.decoder_input(z)

        return self.decode(z)


class DCTHVAE(HVAE):
    """Conditional hierarchical VAE with DCT reconstruction."""

    def __init__(self, ks: list[int] = None, **kwargs):
        super().__init__(**kwargs)
        if ks is None:
            # distribute k between 64 and 1 evenly
            ks = [int(64 / (self.levels - 1) * i) for i in range(self.levels)]

        assert (
            len(ks) == self.num_levels
        ), f"Invalid length: ks should have {self.num_levels} elements."
        self.ks = ks

    def step(self, batch):
        x, y = batch
        losses = []
        for i, k in enumerate(self.ks):
            x_dct = reconstruct_dct(x, k=k).to(self.device)
            outputs = self.forward(x_dct, y, level=i)
            outputs["x"] = x_dct
            losses.append(self.loss_function(**outputs))
        loss = {k: sum(loss[k] for loss in losses) for k in losses[0].keys()}
        return loss, outputs["x_hat"]
