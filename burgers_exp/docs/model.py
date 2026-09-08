"""
model.py

DeepSets encoder + conditional RealNVP normalizing flow.

Expected shapes
---------------
Input:
    x : (B, N_sample_in, Nx)

Target / generated output:
    y : (B, N_sample_out, Ny)

The DeepSets encoder maps each empirical input distribution to one context vector:
    (B, N_sample_in, Nx) -> (B, context_dim)

The conditional normalizing flow models
    p(y | context)

Each output realization is conditionally independent given the DeepSets context.
This makes N_sample_out arbitrary at generation time.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    num_hidden_layers: int = 2,
    activation: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    """Simple MLP helper."""
    layers = []
    dim = in_dim

    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(activation())
        dim = hidden_dim

    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class DeepSetsEncoder(nn.Module):
    """
    Permutation-invariant encoder for a set/batch of stochastic-process samples.

    Input
    -----
    x: (B, N_sample, Nx)

    Output
    ------
    context: (B, context_dim)

    Architecture
    ------------
        h_i = phi(x_i)
        h   = aggregation_i h_i
        c   = rho(h)

    Supported aggregations:
        "mean", "sum", "mean_std"
    """

    def __init__(
        self,
        input_dim: int,
        sample_embed_dim: int = 128,
        context_dim: int = 128,
        hidden_dim: int = 256,
        phi_layers: int = 2,
        rho_layers: int = 2,
        aggregation: str = "mean_std",
    ):
        super().__init__()

        if aggregation not in {"mean", "sum", "mean_std"}:
            raise ValueError(
                f"aggregation must be one of mean/sum/mean_std, got {aggregation}"
            )

        self.input_dim = input_dim
        self.sample_embed_dim = sample_embed_dim
        self.context_dim = context_dim
        self.aggregation = aggregation

        self.phi = build_mlp(
            in_dim=input_dim,
            out_dim=sample_embed_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=phi_layers,
        )

        aggregated_dim = (
            2 * sample_embed_dim if aggregation == "mean_std" else sample_embed_dim
        )

        self.rho = build_mlp(
            in_dim=aggregated_dim,
            out_dim=context_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=rho_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected x shape (B, N_sample, Nx), got {tuple(x.shape)}"
            )

        h = self.phi(x)  # (B, N_sample, sample_embed_dim)

        if self.aggregation == "mean":
            pooled = h.mean(dim=1)

        elif self.aggregation == "sum":
            pooled = h.sum(dim=1)

        else:
            mean = h.mean(dim=1)

            # population std; stable for small sample counts
            var = (h - mean.unsqueeze(1)).pow(2).mean(dim=1)
            std = torch.sqrt(var + 1e-6)

            pooled = torch.cat([mean, std], dim=-1)

        return self.rho(pooled)


class ConditionalAffineCoupling(nn.Module):
    """
    Conditional affine coupling layer used by RealNVP.

    For dimensions selected by the mask:
        y_masked = x_masked

    For remaining dimensions:
        y = x * exp(s) + t

    where s and t depend on:
        masked x + conditioning context.
    """

    def __init__(
        self,
        data_dim: int,
        context_dim: int,
        hidden_dim: int,
        mask: torch.Tensor,
        num_hidden_layers: int = 2,
        scale_limit: float = 2.0,
    ):
        super().__init__()

        if mask.shape != (data_dim,):
            raise ValueError(
                f"mask must have shape ({data_dim},), got {tuple(mask.shape)}"
            )

        self.data_dim = data_dim
        self.context_dim = context_dim
        self.scale_limit = scale_limit

        self.register_buffer("mask", mask.float())

        self.net = build_mlp(
            in_dim=data_dim + context_dim,
            out_dim=2 * data_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

        # Initialize final layer close to identity.
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _scale_shift(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask.view(1, -1)
        x_masked = x * mask

        h = torch.cat([x_masked, context], dim=-1)
        s, t = self.net(h).chunk(2, dim=-1)

        # Bounded log-scale substantially improves numerical stability.
        s = self.scale_limit * torch.tanh(s / self.scale_limit)

        inv_mask = 1.0 - mask
        s = s * inv_mask
        t = t * inv_mask

        return s, t

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Data -> latent direction.

        Returns
        -------
        z:
            transformed tensor, shape (N, data_dim)

        log_det:
            log |det(dz/dx)|, shape (N,)
        """
        mask = self.mask.view(1, -1)

        s, t = self._scale_shift(x, context)

        y = x * mask + (1.0 - mask) * (x * torch.exp(s) + t)

        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(
        self,
        y: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Latent -> data direction.

        Returns
        -------
        x:
            inverse-transformed tensor

        log_det:
            log |det(dx/dy)|
        """
        mask = self.mask.view(1, -1)

        # The masked dimensions are unchanged, so scale/shift can be
        # reconstructed directly from y.
        s, t = self._scale_shift(y, context)

        x = y * mask + (1.0 - mask) * ((y - t) * torch.exp(-s))

        log_det = -s.sum(dim=-1)
        return x, log_det


class ConditionalRealNVP(nn.Module):
    """
    Conditional RealNVP normalizing flow in R^data_dim.

    The base distribution is isotropic standard Gaussian.
    """

    def __init__(
        self,
        data_dim: int,
        context_dim: int,
        num_flow_layers: int = 8,
        hidden_dim: int = 256,
        coupling_hidden_layers: int = 2,
        scale_limit: float = 2.0,
    ):
        super().__init__()

        if data_dim < 2:
            raise ValueError("RealNVP requires data_dim >= 2.")

        self.data_dim = data_dim
        self.context_dim = context_dim

        layers = []

        # Alternating binary masks.
        base_mask = (torch.arange(data_dim) % 2).float()

        for i in range(num_flow_layers):
            mask = base_mask if i % 2 == 0 else 1.0 - base_mask

            layers.append(
                ConditionalAffineCoupling(
                    data_dim=data_dim,
                    context_dim=context_dim,
                    hidden_dim=hidden_dim,
                    mask=mask.clone(),
                    num_hidden_layers=coupling_hidden_layers,
                    scale_limit=scale_limit,
                )
            )

        self.layers = nn.ModuleList(layers)

        self.register_buffer(
            "_log_2pi",
            torch.tensor(2.0 * torch.pi).log(),
        )

    def to_latent(
        self,
        y: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map data y to latent z.

        y:
            (N, data_dim)
        context:
            (N, context_dim)
        """
        z = y
        total_log_det = torch.zeros(
            y.shape[0],
            device=y.device,
            dtype=y.dtype,
        )

        for layer in self.layers:
            z, log_det = layer(z, context)
            total_log_det = total_log_det + log_det

        return z, total_log_det

    def from_latent(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map latent z to generated data y.
        """
        y = z
        total_log_det = torch.zeros(
            z.shape[0],
            device=z.device,
            dtype=z.dtype,
        )

        for layer in reversed(self.layers):
            y, log_det = layer.inverse(y, context)
            total_log_det = total_log_det + log_det

        return y, total_log_det

    def log_prob(
        self,
        y: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conditional log likelihood log p(y | context).

        Returns shape:
            (N,)
        """
        z, log_det = self.to_latent(y, context)

        log_base = -0.5 * (
            z.pow(2) + self._log_2pi.to(dtype=z.dtype)
        ).sum(dim=-1)

        return log_base + log_det

    def sample(
        self,
        context: torch.Tensor,
        num_samples: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples.

        Parameters
        ----------
        context:
            (B, context_dim)

        num_samples:
            number of generated samples for every input distribution

        temperature:
            standard deviation multiplier for the Gaussian base distribution.

        Returns
        -------
        y:
            (B, num_samples, data_dim)
        """
        if context.ndim != 2:
            raise ValueError(
                f"context must have shape (B, context_dim), "
                f"got {tuple(context.shape)}"
            )

        B = context.shape[0]

        z = torch.randn(
            B,
            num_samples,
            self.data_dim,
            device=context.device,
            dtype=context.dtype,
        )
        z = temperature * z

        expanded_context = (
            context[:, None, :]
            .expand(B, num_samples, self.context_dim)
            .reshape(B * num_samples, self.context_dim)
        )

        z_flat = z.reshape(B * num_samples, self.data_dim)

        y_flat, _ = self.from_latent(z_flat, expanded_context)

        return y_flat.reshape(B, num_samples, self.data_dim)


class DeepSetConditionalFlow(nn.Module):
    """
    Full model:
        input empirical stochastic process
            -> DeepSets context
            -> conditional RealNVP distribution

    Training
    --------
    loss = model.nll(x, y)

    Sampling
    --------
    y_pred = model.sample(x, num_samples=N)

    Shapes
    ------
    x:
        (B, N_sample_in, Nx)

    y:
        (B, N_sample_out, Ny)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sample_embed_dim: int = 128,
        context_dim: int = 128,
        encoder_hidden_dim: int = 256,
        encoder_phi_layers: int = 2,
        encoder_rho_layers: int = 2,
        aggregation: str = "mean_std",
        num_flow_layers: int = 8,
        flow_hidden_dim: int = 256,
        coupling_hidden_layers: int = 2,
        scale_limit: float = 2.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        self.encoder = DeepSetsEncoder(
            input_dim=input_dim,
            sample_embed_dim=sample_embed_dim,
            context_dim=context_dim,
            hidden_dim=encoder_hidden_dim,
            phi_layers=encoder_phi_layers,
            rho_layers=encoder_rho_layers,
            aggregation=aggregation,
        )

        self.flow = ConditionalRealNVP(
            data_dim=output_dim,
            context_dim=context_dim,
            num_flow_layers=num_flow_layers,
            hidden_dim=flow_hidden_dim,
            coupling_hidden_layers=coupling_hidden_layers,
            scale_limit=scale_limit,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an input sample set into its distribution-level context."""
        return self.encoder(x)

    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log p(y_j | input sample set).

        Parameters
        ----------
        x:
            (B, N_sample_in, Nx)

        y:
            (B, N_sample_out, Ny)

        Returns
        -------
        log_prob:
            (B, N_sample_out)
        """
        if y.ndim != 3:
            raise ValueError(
                f"Expected y shape (B, N_sample, Ny), got {tuple(y.shape)}"
            )

        B, N, Ny = y.shape

        if Ny != self.output_dim:
            raise ValueError(
                f"Expected Ny={self.output_dim}, got Ny={Ny}"
            )

        if x.shape[0] != B:
            raise ValueError(
                f"x and y batch sizes differ: {x.shape[0]} vs {B}"
            )

        context = self.encode(x)  # (B, context_dim)

        expanded_context = (
            context[:, None, :]
            .expand(B, N, self.context_dim)
            .reshape(B * N, self.context_dim)
        )

        y_flat = y.reshape(B * N, Ny)

        log_prob_flat = self.flow.log_prob(
            y_flat,
            expanded_context,
        )

        return log_prob_flat.reshape(B, N)

    def nll(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Negative conditional log likelihood.

        reduction:
            "mean", "sum", or "none"
        """
        loss = -self.log_prob(x, y)

        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss

        raise ValueError(
            f"reduction must be mean/sum/none, got {reduction}"
        )

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        num_samples: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate target-process realizations.

        Parameters
        ----------
        x:
            (B, N_sample_in, Nx)

        num_samples:
            desired output sample count. If None, use N_sample_in.

        Returns
        -------
        y:
            (B, num_samples, Ny)
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected x shape (B, N_sample, Nx), got {tuple(x.shape)}"
            )

        if num_samples is None:
            num_samples = x.shape[1]

        context = self.encode(x)

        return self.flow.sample(
            context=context,
            num_samples=num_samples,
            temperature=temperature,
        )

    def forward(
        self,
        x: torch.Tensor,
        num_samples: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Convenience forward method for generation.

        During training, prefer:
            loss = model.nll(x, y)
        """
        context = self.encode(x)

        if num_samples is None:
            num_samples = x.shape[1]

        return self.flow.sample(
            context=context,
            num_samples=num_samples,
            temperature=temperature,
        )


if __name__ == "__main__":
    # Minimal shape test.
    torch.manual_seed(0)

    B = 4
    N_sample = 32
    Nx = 64
    Ny = 48

    model = DeepSetConditionalFlow(
        input_dim=Nx,
        output_dim=Ny,
        sample_embed_dim=128,
        context_dim=128,
        num_flow_layers=8,
        flow_hidden_dim=256,
    )

    x = torch.randn(B, N_sample, Nx)
    y = torch.randn(B, N_sample, Ny)

    loss = model.nll(x, y)
    print("NLL:", loss.item())

    loss.backward()

    with torch.no_grad():
        y_generated = model.sample(
            x,
            num_samples=100,
        )

    print("input shape:    ", x.shape)
    print("target shape:   ", y.shape)
    print("generated shape:", y_generated.shape)
