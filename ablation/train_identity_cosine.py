pass





from __future__ import annotations

import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import ablation.fedpretrain as original


class IdentityCosineFederatedGFM(original.FederatedGFM):
    pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = float(
            OmegaConf.select(self.cfg, "ablation.temperature", default=0.1)
        )
        if self.temperature <= 0:
            raise ValueError("ablation.temperature must be positive")
        self._initialize_cali_as_identity()

    def _initialize_cali_as_identity(self) -> None:
        pass
        final_linear = self.de.dm_cali.blocks[-1].lin
        width = int(self.cfg.Cali.aligned_feat_dim)
        with torch.no_grad():
            final_linear.weight.zero_()
            final_linear.bias.zero_()
            if bool(self.cfg.Cali.softplus):
                identity_logit = math.log(math.expm1(1.0))
                final_linear.bias[:width].fill_(identity_logit)
                final_linear.bias[2 * width : 3 * width].fill_(identity_logit)
            else:
                final_linear.bias[:width].fill_(1.0)
                final_linear.bias[2 * width : 3 * width].fill_(1.0)

    def on_train_epoch_start(self):
        
        return None

    def configure_optimizers(self):
        
        
        return torch.optim.AdamW(
            [
                {"params": self.GNNEnc.parameters(), "lr": self.cfg.PTModel.lr},
                {
                    "params": self.de.dm_cali.parameters(),
                    "lr": self.cfg.PTModel.lr * 0.5,
                },
            ],
            weight_decay=self.cfg.PTModel.weight_decay,
        )

    def _episode_batch_step(self, batch, training: bool):
        if not isinstance(batch, dict):
            raise TypeError(
                "IdentityCosineFederatedGFM requires the mapping batch produced "
                "by fedpretrain.GPUEpisodeLoader"
            )
        if self.domain_embeddings is None:
            
            
            self._compute_domain_embeddings()

        embeddings = self.domain_embeddings.detach()
        gamma_f, beta_f, gamma_l, beta_l = self.de.dm_cali(embeddings)
        self.gamma_f, self.beta_f = gamma_f, beta_f
        self.gamma_l, self.beta_l = gamma_l, beta_l

        graph = self.comb_pretrained_graphs
        hidden, _ = self.forward_backbone(
            graph.x,
            graph.edge_index,
            graph.xe if hasattr(graph, "xe") else None,
            graph.batch,
        )

        support = batch["support"]
        query = batch["query"]
        classes = batch["classes"]
        graph_ids = batch["graph"]
        batch_size = support.size(0)
        num_classes = classes.size(1)
        if support.size(1) % num_classes or query.size(1) % num_classes:
            raise ValueError("Support/query counts must be balanced by class")
        shots = support.size(1) // num_classes
        queries_per_class = query.size(1) // num_classes

        gamma = gamma_f[graph_ids].unsqueeze(1)
        beta = beta_f[graph_ids].unsqueeze(1)
        z_support = gamma * hidden[support] + beta
        z_query = gamma * hidden[query] + beta

        
        
        z_support = F.normalize(z_support, p=2, dim=-1)
        z_query = F.normalize(z_query, p=2, dim=-1)
        prototypes = z_support.reshape(
            batch_size, num_classes, shots, z_support.size(-1)
        ).mean(dim=2)
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        logits = torch.einsum("bqh,bmh->bqm", z_query, prototypes)
        logits = logits / self.temperature

        targets = torch.arange(num_classes, device=self.device)
        targets = targets.repeat_interleave(queries_per_class)
        targets = targets.unsqueeze(0).expand(batch_size, -1)
        loss = F.cross_entropy(
            logits.reshape(-1, num_classes),
            targets.reshape(-1),
            label_smoothing=0.1 if training else 0.0,
        )
        accuracy = (logits.argmax(dim=-1) == targets).float().mean()
        return loss, accuracy


def main() -> None:
    defaults = {
        "+ablation.objective=cosine_prototype": "ablation.objective=",
        "+ablation.cali_mode=trainable_identity": "ablation.cali_mode=",
        "+ablation.temperature=0.1": "ablation.temperature=",
    }
    for default, key in defaults.items():
        if not any(key in argument for argument in sys.argv[1:]):
            sys.argv.append(default)
    original.FederatedGFM = IdentityCosineFederatedGFM
    original.main()


if __name__ == "__main__":
    main()

