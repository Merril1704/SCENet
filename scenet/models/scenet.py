from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SCENetOutputs:
    logits: torch.Tensor
    gates: torch.Tensor
    selected: torch.Tensor
    concepts: torch.Tensor


class SCENet(nn.Module):
    """Structured Concept-Embedded Network (SCENet) for tabular data.

    Architecture (binary classification):
      x -> feature gates g -> selected features z = x*g -> concept layer c -> output logits

    Notes:
    - Uses per-feature *local* gates: g_i = sigmoid(a_i * x_i + b_i)
    - Provides intrinsic interpretability via gates and concept bottleneck.
    """

    def __init__(
        self,
        *,
        n_features: int,
        n_concepts: int = 16,
        concept_activation: str = "relu",
        dropout: float = 0.0,
        gate_type: str = "sigmoid",
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if n_features <= 0:
            raise ValueError("n_features must be > 0")
        if n_concepts <= 0:
            raise ValueError("n_concepts must be > 0")

        self.n_features = int(n_features)
        self.n_concepts = int(n_concepts)

        gate_type = str(gate_type).lower().strip()
        if gate_type not in {"sigmoid", "hard_concrete"}:
            raise ValueError(f"Unsupported gate_type: {gate_type}")
        self.gate_type = gate_type

        gate_temperature = float(gate_temperature)
        if gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be > 0")
        self.gate_temperature = gate_temperature

        # Hard-concrete (L0) gate hyperparameters (Louizos et al., 2018).
        # Kept as fixed defaults; exposed mainly via gate_type.
        self.hard_concrete_beta = 2.0 / 3.0
        self.hard_concrete_gamma = -0.1
        self.hard_concrete_zeta = 1.1

        # Per-feature gate parameters.
        self.gate_scale = nn.Parameter(torch.zeros(self.n_features))
        self.gate_bias = nn.Parameter(torch.zeros(self.n_features))

        act: nn.Module
        if concept_activation == "relu":
            act = nn.ReLU()
        elif concept_activation == "tanh":
            act = nn.Tanh()
        elif concept_activation == "sigmoid":
            act = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported concept_activation: {concept_activation}")

        self.concept = nn.Sequential(
            nn.Linear(self.n_features, self.n_concepts),
            act,
            nn.Dropout(p=float(dropout)),
        )
        self.out = nn.Linear(self.n_concepts, 1)

    def forward(self, x: torch.Tensor) -> SCENetOutputs:
        # x: (batch, n_features)
        gate_logits = (x * self.gate_scale + self.gate_bias) / float(self.gate_temperature)

        if self.gate_type == "sigmoid":
            g = torch.sigmoid(gate_logits)
        else:
            # Hard-concrete stochastic gates during training; deterministic during eval.
            if self.training:
                u = torch.rand_like(gate_logits)
                s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + gate_logits) / float(self.hard_concrete_beta))
            else:
                s = torch.sigmoid(gate_logits)

            s_bar = s * (float(self.hard_concrete_zeta) - float(self.hard_concrete_gamma)) + float(
                self.hard_concrete_gamma
            )
            g = torch.clamp(s_bar, min=0.0, max=1.0)
        z = x * g
        c = self.concept(z)
        logits = self.out(c).squeeze(-1)
        return SCENetOutputs(logits=logits, gates=g, selected=z, concepts=c)

    @property
    def concept_weight(self) -> torch.Tensor:
        # First layer is Linear
        layer = self.concept[0]
        assert isinstance(layer, nn.Linear)
        return layer.weight
