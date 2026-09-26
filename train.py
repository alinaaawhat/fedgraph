pass





from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
import torch.nn.functional as F

import ablation.fedpretrain as original
import ablation.train_v3 as v3


class BatchedSQV(nn.Module):
    pass

    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = int(cfg.EditGrad.hidden_dim)
        self.heads = int(OmegaConf.select(cfg, "PAMA.heads", default=1))
        self.head_dim = int(
            OmegaConf.select(cfg, "PAMA.d_attn", default=self.hidden_dim)
        )
        self.projected_dim = self.heads * self.head_dim
        self.temperature = float(
            OmegaConf.select(cfg, "ablation.sqv_temperature", default=1.0)
        )
        if self.temperature <= 0:
            raise ValueError("ablation.sqv_temperature must be positive")
        dropout = float(OmegaConf.select(cfg, "ablation.sqv_dropout", default=0.0))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.query = nn.Linear(self.hidden_dim, self.projected_dim)
        self.key = nn.Linear(self.hidden_dim, self.projected_dim)
        self._identity_initialization()

    def _identity_initialization(self):
        if self.heads != 1 or self.head_dim != self.hidden_dim:
            return
        with torch.no_grad():
            self.query.weight.copy_(torch.eye(self.hidden_dim))
            self.key.weight.copy_(torch.eye(self.hidden_dim))
            self.query.bias.zero_()
            self.key.bias.zero_()

    def forward(self, z_query, z_support, support_targets, num_classes):
        if z_query.dim() == 2:
            z_query = z_query.unsqueeze(0)
            z_support = z_support.unsqueeze(0)
            support_targets = support_targets.unsqueeze(0)
            squeeze = True
        elif z_query.dim() == 3:
            squeeze = False
        else:
            raise ValueError("z_query must have shape [Q,H] or [B,Q,H]")
        if z_support.dim() != 3 or support_targets.dim() != 2:
            raise ValueError(
                "z_support/support_targets must have shapes [B,S,H]/[B,S]"
            )
        if z_query.size(0) != z_support.size(0) or (
            z_support.size(0) != support_targets.size(0)
        ):
            raise ValueError("SQV batch dimensions do not match")
        if z_support.size(1) != support_targets.size(1):
            raise ValueError("Every support key must have one label value")

        batch_size, num_queries, _ = z_query.shape
        num_support = z_support.size(1)
        q = self.query(self.norm(z_query)).view(
            batch_size, num_queries, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        k = self.key(self.norm(z_support)).view(
            batch_size, num_support, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5 * self.temperature)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        values = F.one_hot(
            support_targets.long(), num_classes=int(num_classes)
        ).to(dtype=attention.dtype)
        probability = torch.einsum("bhqs,bsm->bhqm", attention, values)
        probability = probability.mean(dim=1).clamp_min(1e-8)
        logits = probability.log()
        return logits.squeeze(0) if squeeze else logits


class V4FederatedGFM(v3.V3FederatedGFM):
    pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sqv = BatchedSQV(self.cfg)
        self.use_fixed_domain_state = bool(
            OmegaConf.select(
                self.cfg, "ablation.fixed_domain_state", default=False
            )
        )
        if self.use_fixed_domain_state:
            self.shared_domain_state = nn.Parameter(torch.zeros(
                1, int(self.cfg.EditGrad.compressed_dim)
            ))
        else:
            self.register_parameter("shared_domain_state", None)

    def configure_optimizers(self):
        groups = [
            {"params": self.GNNEnc.parameters(), "lr": self.cfg.PTModel.lr},
            {
                "params": self.de.dm_cali.parameters(),
                "lr": self.cfg.PTModel.lr * 0.5,
            },
            {"params": self.sqv.parameters(), "lr": self.cfg.PTModel.lr},
        ]
        if self.shared_domain_state is not None:
            groups.append({
                "params": [self.shared_domain_state],
                "lr": self.cfg.PTModel.lr * 0.5,
            })
        return torch.optim.AdamW(
            groups,
            weight_decay=self.cfg.PTModel.weight_decay,
        )

    def _domain_codes(self, batch_size, training):
        if self.shared_domain_state is not None:
            return self.shared_domain_state.expand(batch_size, -1)
        return super()._domain_codes(batch_size, training)

    def _episode_batch_step(self, batch, training):
        if not isinstance(batch, dict):
            raise TypeError("V4 requires GPUEpisodeLoader mapping batches")
        if self.domain_embeddings is None:
            self._compute_domain_embeddings()
        graph = self.comb_pretrained_graphs
        hidden, _ = self.forward_backbone(
            graph.x,
            graph.edge_index,
            graph.xe if hasattr(graph, "xe") else None,
            graph.batch,
        )
        support, query = batch["support"], batch["query"]
        batch_size = int(support.size(0))
        num_classes = int(batch["classes"].size(1))
        if support.size(1) % num_classes or query.size(1) % num_classes:
            raise ValueError("SQV requires class-balanced support/query episodes")
        shots = support.size(1) // num_classes
        queries_per_class = query.size(1) // num_classes
        support_targets = torch.arange(
            num_classes, device=self.device
        ).repeat_interleave(shots).unsqueeze(0).expand(batch_size, -1)
        query_targets = torch.arange(
            num_classes, device=self.device
        ).repeat_interleave(queries_per_class).unsqueeze(0).expand(batch_size, -1)

        base_logits = self.sqv(
            hidden[query], hidden[support], support_targets, num_classes
        )
        base_loss = F.cross_entropy(
            base_logits.reshape(-1, num_classes),
            query_targets.reshape(-1),
            label_smoothing=0.1 if training else 0.0,
        )
        loss = base_loss
        cali_logits = None
        lambda_cali = float(
            OmegaConf.select(self.cfg, "ablation.lambda_cali_cls", default=0.5)
        )
        if lambda_cali > 0:
            codes = self._domain_codes(batch_size, training)
            delta_gamma, delta_beta = self._cali_residual(codes)
            cali_support = self._transform(
                hidden, support, delta_gamma, delta_beta
            )
            cali_query = self._transform(hidden, query, delta_gamma, delta_beta)
            cali_logits = self.sqv(
                cali_query, cali_support, support_targets, num_classes
            )
            cali_loss = F.cross_entropy(
                cali_logits.reshape(-1, num_classes),
                query_targets.reshape(-1),
                label_smoothing=0.1 if training else 0.0,
            )
            consistency_weight = float(
                OmegaConf.select(
                    self.cfg, "ablation.lambda_consistency", default=0.0
                )
            )
            consistency = F.kl_div(
                F.log_softmax(cali_logits, dim=-1),
                F.softmax(base_logits.detach(), dim=-1),
                reduction="batchmean",
            )
            loss = loss + lambda_cali * cali_loss + consistency_weight * consistency
        logits = cali_logits if cali_logits is not None else base_logits
        accuracy = (
            logits.argmax(dim=-1) == query_targets
        ).float().mean()
        return loss, accuracy


def _add_v4_metadata(state):
    state = v3._add_v3_metadata(state)
    fixed = bool(OmegaConf.select(
        state.get("config"), "ablation.fixed_domain_state", default=False
    ))
    state["v4_classifier"] = {
        "name": "support_query_value_attention",
        "query": "node representation",
        "key": "support node representation",
        "value": "episode-local one-hot support label",
        "source_global_label_values": False,
    }
    state["fixed_domain_state_ablation"] = {
        "enabled": fixed,
        "state": "shared_trainable_parameter" if fixed else None,
        "graph_specific": not fixed,
    }
    return state


def _augment_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    torch.save(_add_v4_metadata(state), path)


@hydra.main(
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="fed-main",
    version_base=None,
)
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "ablation.variant", "E", force_add=True)
    v3._configure_variant(cfg)
    cfg.EditGrad.DE_type = "conv"
    original._compute_shared_editgrad = v3.compute_shared_editgrad_v3
    v3.install_shared_editgrad_v3.original_install = original._install_shared_editgrad
    original._install_shared_editgrad = v3.install_shared_editgrad_v3
    original.FederatedGFM = V4FederatedGFM
    original._run_local_training = v3.run_local_training_v3

    real_save = torch.save
    tracker = v3.MovingAverageBestTracker(
        real_save,
        int(OmegaConf.select(cfg, "ablation.best_window", default=5)),
    )
    original.torch.save = tracker
    try:
        _, final_path = original.main.__wrapped__.__wrapped__(cfg)
    finally:
        original.torch.save = real_save
    _augment_checkpoint(final_path)

    if tracker.best_round_path is not None:
        final = torch.load(final_path, map_location="cpu", weights_only=False)
        best = torch.load(
            tracker.best_round_path, map_location="cpu", weights_only=False
        )
        state = best["global_state_dict"]
        final["model_state_dict"] = state
        final["backbone_state_dict"] = original._extract_prefixed_state(
            state, "GNNEnc."
        )
        final["frozen_backbone_state_dict"] = original._extract_prefixed_state(
            state, "de.frozen_backboneGNN."
        )
        final["selected_round"] = tracker.best_round
        final["selected_moving_val_acc"] = tracker.best_metric
        best_path = os.path.join(
            os.path.dirname(final_path), "best_federated_gfm_model.pt"
        )
        real_save(_add_v4_metadata(final), best_path)
        original.logger.info(
            "V4 best SQV model: round=%d metric=%.4f path=%s",
            tracker.best_round,
            tracker.best_metric,
            best_path,
        )


if __name__ == "__main__":
    defaults = {
        "+ablation.sqv_temperature=1.0": "ablation.sqv_temperature=",
        "+ablation.sqv_dropout=0.0": "ablation.sqv_dropout=",
        "+ablation.fixed_domain_state=false": "ablation.fixed_domain_state=",
        "+ablation.centered_cali=true": "ablation.centered_cali=",
        "+ablation.multi_episode=true": "ablation.multi_episode=",
        "+ablation.sample_domain_responses=true": "ablation.sample_domain_responses=",
        "+ablation.editgrad_bank_size=32": "ablation.editgrad_bank_size=",
        "+ablation.editgrad_k_shot=5": "ablation.editgrad_k_shot=",
        "+ablation.editgrad_m_way=5": "ablation.editgrad_m_way=",
        "+ablation.diversity_weight=0.001": "ablation.diversity_weight=",
        "+ablation.stability_weight=0.01": "ablation.stability_weight=",
        "+ablation.projection_patience=40": "ablation.projection_patience=",
        "+ablation.cali_scale=0.1": "ablation.cali_scale=",
        "+ablation.lambda_cali_cls=0.5": "ablation.lambda_cali_cls=",
        "+ablation.lambda_consistency=0.0": "ablation.lambda_consistency=",
        "+ablation.lambda_rank=0.0": "ablation.lambda_rank=",
        "+ablation.best_window=5": "ablation.best_window=",
        "Cali.dropout=0.0": "Cali.dropout=",
        "PTModel.lr=0.001": "PTModel.lr=",
        "federated.local_epochs=3": "federated.local_epochs=",
        "federated.rounds=700": "federated.rounds=",
        "pretrain.t_query=5": "pretrain.t_query=",
        "EditGrad.n_layers=2": "EditGrad.n_layers=",
        "EditGrad.n_layers_editgrad=1": "EditGrad.n_layers_editgrad=",
    }
    for default, key in defaults.items():
        if not any(key in argument for argument in sys.argv[1:]):
            sys.argv.append(default)
    main()
