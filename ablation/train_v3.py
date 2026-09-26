pass





from __future__ import annotations

import copy
from collections import deque
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F

import ablation.fedpretrain as original
from ablation.train_identity_cosine import IdentityCosineFederatedGFM
from model.editgrad import DomainEmbedder


DOMAIN_RESPONSE_STATE = {}


def _class_balanced_supports(graph, count, k_shot, m_way, seed):
    classes = graph.y.unique(sorted=True)
    m_way = min(int(m_way), int(classes.numel()))
    pools = [(graph.y == label).nonzero(as_tuple=False).view(-1) for label in classes]
    generator = torch.Generator(device=graph.x.device).manual_seed(int(seed))
    supports = []
    for _ in range(int(count)):
        positions = torch.randperm(
            classes.numel(), generator=generator, device=graph.x.device
        )[:m_way]
        chosen = []
        for position in positions:
            pool = pools[int(position)]
            take = min(int(k_shot), int(pool.numel()))
            order = torch.randperm(
                pool.numel(), generator=generator, device=graph.x.device
            )[:take]
            chosen.append(pool[order])
        supports.append(torch.cat(chosen))
    return supports


def _episode_gradient_responses(extractor, graph, supports):
    pass
    model = extractor.frozen_backbone.to(extractor.device).eval()
    if extractor._theta0 is None:
        extractor._theta0 = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    model.load_state_dict(extractor._theta0, strict=False)
    graph = graph.to(extractor.device)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits, _ = model(graph)
        selected = next(
            parameter
            for name, parameter in model.named_parameters()
            if "weight" in name and parameter.requires_grad
        )
        responses = []
        for index, support in enumerate(supports):
            loss = F.cross_entropy(logits[support], graph.y[support])
            gradient = torch.autograd.grad(
                loss,
                selected,
                retain_graph=index < len(supports) - 1,
            )[0]
            if gradient.shape[0] != graph.x.shape[1]:
                gradient = gradient.T
            responses.append(
                (-float(extractor.cfg.EditGrad.probe_lr) * gradient)
                .detach()
                .cpu()
            )
    return responses


def _normalized_distance_matrix(rows):
    rows = F.normalize(rows, p=2, dim=-1)
    distances = torch.cdist(rows, rows)
    mask = ~torch.eye(rows.size(0), dtype=torch.bool, device=rows.device)
    return distances / distances[mask].mean().clamp_min(1e-8)


def _fit_hierarchical_projection(extractor, padded_responses, slices, cfg, device):
    pass
    matrices = [response.to(device) for response in padded_responses]
    raw_rows = torch.stack([matrix.reshape(-1) for matrix in matrices])
    raw_centers = torch.stack([raw_rows[s].mean(dim=0) for s in slices])
    target_center_distances = _normalized_distance_matrix(raw_centers)
    diversity_weight = float(
        OmegaConf.select(cfg, "ablation.diversity_weight", default=1e-3)
    )
    stability_weight = float(
        OmegaConf.select(cfg, "ablation.stability_weight", default=1e-2)
    )
    patience = int(
        OmegaConf.select(cfg, "ablation.projection_patience", default=40)
    )
    optimizer = torch.optim.Adam(
        extractor.projection.parameters(), lr=float(cfg.EditGrad.DE.train_lr)
    )
    best_state = copy.deepcopy(extractor.projection.state_dict())
    best_objective, stale = float("inf"), 0
    epochs = int(cfg.EditGrad.DE.train_epochs)
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        embeddings = torch.stack([
            F.normalize(extractor.projection(matrix), p=2, dim=-1)
            for matrix in matrices
        ])
        centers = torch.stack([
            F.normalize(embeddings[s].mean(dim=0), p=2, dim=-1)
            for s in slices
        ])
        center_distance = F.mse_loss(
            _normalized_distance_matrix(centers), target_center_distances
        )
        gram = centers @ centers.T
        center_diversity = -torch.logdet(
            gram
            + 1e-6 * torch.eye(
                gram.size(0), device=device, dtype=gram.dtype
            )
        )
        within_stability = torch.stack([
            (embeddings[s] - centers[index]).square().sum(dim=-1).mean()
            for index, s in enumerate(slices)
        ]).mean()
        objective = (
            center_distance
            + diversity_weight * center_diversity
            + stability_weight * within_stability
        )
        objective.backward()
        optimizer.step()
        value = float(objective.detach())
        if value < best_objective - 1e-7:
            best_objective, stale = value, 0
            best_state = copy.deepcopy(extractor.projection.state_dict())
        else:
            stale += 1
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"Projection {epoch:03d}/{epochs - 1:03d} | total={value:.6f} "
                f"center_dist={float(center_distance.detach()):.6f} "
                f"center_div={float(center_diversity.detach()):.6f} "
                f"within={float(within_stability.detach()):.6f}"
            )
        if patience > 0 and stale >= patience:
            print(f"Projection early stop at epoch {epoch}; restoring best")
            break
    extractor.projection.load_state_dict(best_state)
    extractor.projection.eval()
    with torch.no_grad():
        embeddings = torch.stack([
            F.normalize(extractor.projection(matrix), p=2, dim=-1)
            for matrix in matrices
        ])
        centers = torch.stack([
            F.normalize(embeddings[s].mean(dim=0), p=2, dim=-1)
            for s in slices
        ])
    return embeddings, centers


def compute_shared_editgrad_v3(cfg, clients, frozen_backbone, device):
    pass
    shared_cfg = original._shared_editgrad_cfg(cfg)
    shared_cfg.EditGrad.DE_type = "conv"
    embedder = DomainEmbedder(
        frozen_backboneGNN=copy.deepcopy(frozen_backbone).to(device),
        cfg=shared_cfg,
    ).to(device)
    extractor = embedder.dm_extractor
    use_multi = bool(
        OmegaConf.select(cfg, "ablation.multi_episode", default=True)
    )
    bank_size = int(
        OmegaConf.select(cfg, "ablation.editgrad_bank_size", default=32)
    ) if use_multi else 1
    bank_k = int(
        OmegaConf.select(cfg, "ablation.editgrad_k_shot", default=5)
    )
    bank_m = int(
        OmegaConf.select(cfg, "ablation.editgrad_m_way", default=5)
    )

    all_responses, slices = [], []
    for domain_id, graph in enumerate(clients.values()):
        supports = _class_balanced_supports(
            graph,
            bank_size,
            bank_k,
            bank_m,
            int(cfg.pretrain.seed) + 10_000 * domain_id,
        )
        responses = _episode_gradient_responses(extractor, graph, supports)
        start = len(all_responses)
        all_responses.extend(responses)
        slices.append(slice(start, len(all_responses)))

    width = max(int(response.shape[1]) for response in all_responses)
    padded = [
        F.pad(response, (0, width - int(response.shape[1])))
        for response in all_responses
    ]
    embeddings, centers = _fit_hierarchical_projection(
        extractor, padded, slices, shared_cfg, device
    )
    banks = [embeddings[s].detach() for s in slices]
    extractor._embedding_banks = banks
    extractor._e = centers
    extractor._d_c_max = width
    extractor._delta_matrices = [
        torch.stack(padded[s]).mean(dim=0).detach().cpu()
        for s in slices
    ]
    extractor._save_cache()

    reference = centers.mean(dim=0)
    DOMAIN_RESPONSE_STATE.clear()
    DOMAIN_RESPONSE_STATE.update({
        "banks": [bank.detach().cpu() for bank in banks],
        "centers": centers.detach().cpu(),
        "reference": reference.detach().cpu(),
        "source_names": list(clients.keys()),
    })
    original.logger.info(
        "V3 stable domain encoder fitted: clients=%d responses=%d bank_size=%d",
        len(clients), len(all_responses), bank_size,
    )
    return embedder


def install_shared_editgrad_v3(model, shared_embedder, client_index):
    install_shared_editgrad_v3.original_install(
        model, shared_embedder, client_index
    )
    extractor = shared_embedder.dm_extractor
    model.domain_embedding_banks = [
        bank.detach().to(model.device) for bank in extractor._embedding_banks
    ]
    model.domain_centers = extractor._e.detach().to(model.device)
    model.domain_reference = extractor._e.mean(dim=0).detach().to(model.device)
    model.client_domain_index = int(client_index)


def _cosine_logits(z_support, z_query, num_classes, temperature):
    batch_size = z_support.size(0)
    shots = z_support.size(1) // num_classes
    z_support = F.normalize(z_support, p=2, dim=-1)
    z_query = F.normalize(z_query, p=2, dim=-1)
    prototypes = z_support.reshape(
        batch_size, num_classes, shots, z_support.size(-1)
    ).mean(dim=2)
    prototypes = F.normalize(prototypes, p=2, dim=-1)
    return torch.einsum("bqh,bmh->bqm", z_query, prototypes) / temperature


class V3FederatedGFM(IdentityCosineFederatedGFM):
    pass

    domain_embedding_banks = None
    domain_centers = None
    domain_reference = None
    client_domain_index = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cali_scale = float(
            OmegaConf.select(self.cfg, "ablation.cali_scale", default=0.1)
        )
        self.centered_cali = bool(
            OmegaConf.select(self.cfg, "ablation.centered_cali", default=True)
        )

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

    def _domain_codes(self, batch_size, training):
        if self.domain_centers is None:
            code = self.domain_embeddings[0].detach()
            return code.unsqueeze(0).expand(batch_size, -1)
        if training and bool(
            OmegaConf.select(
                self.cfg, "ablation.sample_domain_responses", default=True
            )
        ):
            bank = self.domain_embedding_banks[self.client_domain_index]
            indices = torch.randint(
                bank.size(0), (batch_size,), device=self.device
            )
            return bank[indices]
        code = self.domain_centers[self.client_domain_index]
        return code.unsqueeze(0).expand(batch_size, -1)

    def _cali_residual(self, codes):
        raw_gamma, raw_beta, _, _ = self.de.dm_cali(codes)
        if self.centered_cali:
            reference = self.domain_reference.unsqueeze(0).expand_as(codes)
            ref_gamma, ref_beta, _, _ = self.de.dm_cali(reference)
            raw_gamma = raw_gamma - ref_gamma
            raw_beta = raw_beta - ref_beta
        else:
            raw_gamma = raw_gamma - 1.0
        delta_gamma = self.cali_scale * torch.tanh(raw_gamma)
        delta_beta = self.cali_scale * torch.tanh(raw_beta)
        return delta_gamma, delta_beta

    @staticmethod
    def _transform(hidden, indices, delta_gamma, delta_beta):
        selected = hidden[indices]
        return selected + (
            delta_gamma.unsqueeze(1) * selected + delta_beta.unsqueeze(1)
        )

    def _episode_batch_step(self, batch, training):
        if not isinstance(batch, dict):
            raise TypeError("V3 requires GPUEpisodeLoader mapping batches")
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
        num_classes = int(batch["classes"].size(1))
        batch_size = int(support.size(0))
        queries_per_class = query.size(1) // num_classes
        targets = torch.arange(
            num_classes, device=self.device
        ).repeat_interleave(queries_per_class)
        targets = targets.unsqueeze(0).expand(batch_size, -1)

        base_logits = _cosine_logits(
            hidden[support], hidden[query], num_classes, self.temperature
        )
        base_loss = F.cross_entropy(
            base_logits.reshape(-1, num_classes),
            targets.reshape(-1),
            label_smoothing=0.1 if training else 0.0,
        )
        loss = base_loss

        lambda_cali = float(
            OmegaConf.select(self.cfg, "ablation.lambda_cali_cls", default=0.5)
        )
        cali_logits = None
        if lambda_cali > 0:
            codes = self._domain_codes(batch_size, training)
            delta_gamma, delta_beta = self._cali_residual(codes)
            cali_support = self._transform(
                hidden, support, delta_gamma, delta_beta
            )
            cali_query = self._transform(hidden, query, delta_gamma, delta_beta)
            cali_logits = _cosine_logits(
                cali_support, cali_query, num_classes, self.temperature
            )
            cali_loss = F.cross_entropy(
                cali_logits.reshape(-1, num_classes),
                targets.reshape(-1),
                label_smoothing=0.1 if training else 0.0,
            )
            consistency_weight = float(
                OmegaConf.select(
                    self.cfg, "ablation.lambda_consistency", default=0.1
                )
            )
            consistency = F.kl_div(
                F.log_softmax(cali_logits, dim=-1),
                F.softmax(base_logits.detach(), dim=-1),
                reduction="batchmean",
            )
            loss = loss + lambda_cali * cali_loss + consistency_weight * consistency

            rank_weight = float(
                OmegaConf.select(self.cfg, "ablation.lambda_rank", default=0.0)
            )
            if training and rank_weight > 0 and self.domain_centers.size(0) > 1:
                other_ids = torch.randint(
                    self.domain_centers.size(0) - 1,
                    (batch_size,),
                    device=self.device,
                )
                other_ids += (other_ids >= self.client_domain_index).long()
                wrong_gamma, wrong_beta = self._cali_residual(
                    self.domain_centers[other_ids]
                )
                wrong_logits = _cosine_logits(
                    self._transform(hidden, support, wrong_gamma, wrong_beta),
                    self._transform(hidden, query, wrong_gamma, wrong_beta),
                    num_classes,
                    self.temperature,
                )
                wrong_loss = F.cross_entropy(
                    wrong_logits.reshape(-1, num_classes), targets.reshape(-1)
                )
                margin = float(
                    OmegaConf.select(self.cfg, "ablation.rank_margin", default=0.01)
                )
                loss = loss + rank_weight * F.relu(
                    margin + cali_loss - wrong_loss
                )

        logits_for_metric = cali_logits if cali_logits is not None else base_logits
        accuracy = (
            logits_for_metric.argmax(dim=-1) == targets
        ).float().mean()
        return loss, accuracy


def run_local_training_v3(model, optimizer, train_loader, val_loader, local_epochs):
    pass
    optimizer = model.configure_optimizers()
    model.reset_local_metrics()
    model.train()
    for batch_index, batch in enumerate(train_loader):
        if batch_index >= local_epochs:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_step(batch, batch_index)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    if hasattr(val_loader, "generator"):
        val_loader.generator.manual_seed(int(model.cfg.pretrain.seed) + 1_000_000)
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            model.validation_step(batch, batch_index)
    return {
        name: original._metric_value(value)
        for name, value in model.local_metrics.items()
    }


class MovingAverageBestTracker:
    def __init__(self, real_save, window=5):
        self.real_save = real_save
        self.window = deque(maxlen=int(window))
        self.best_metric = -float("inf")
        self.best_round = None
        self.best_round_path = None

    def __call__(self, obj, path, *args, **kwargs):
        self.real_save(obj, path, *args, **kwargs)
        if not isinstance(obj, dict) or os.path.basename(str(path)) != "last.pt":
            return
        value = obj.get("server_metrics", {}).get("val_acc")
        if value is None:
            return
        self.window.append(float(value))
        moving = sum(self.window) / len(self.window)
        if moving > self.best_metric:
            self.best_metric = moving
            self.best_round = int(obj["round"])
            self.best_round_path = os.path.join(
                os.path.dirname(path), "best_round.pt"
            )
            saved = dict(obj)
            saved["selection_metric"] = "fixed_episode_local_val_moving_average"
            saved["selection_value"] = moving
            self.real_save(saved, self.best_round_path)


def _add_v3_metadata(state):
    state["v3_domain_generalization"] = {
        "kind": "gradient_conditioned_few_shot_dg",
        "banks": DOMAIN_RESPONSE_STATE.get("banks"),
        "centers": DOMAIN_RESPONSE_STATE.get("centers"),
        "reference": DOMAIN_RESPONSE_STATE.get("reference"),
        "source_names": DOMAIN_RESPONSE_STATE.get("source_names"),
        "target_updates": 0,
    }
    return state


def _augment_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    torch.save(_add_v3_metadata(state), path)


def _configure_variant(cfg):
    pass
    variant = str(
        OmegaConf.select(cfg, "ablation.variant", default="E")
    ).upper()
    settings = {
        "A": {"lambda_cali_cls": 0.0, "centered_cali": True, "multi_episode": False},
        "B": {"lambda_cali_cls": 0.5, "centered_cali": False, "multi_episode": False},
        "C": {"lambda_cali_cls": 0.5, "centered_cali": True, "multi_episode": False},
        "D": {"lambda_cali_cls": 0.5, "centered_cali": True, "multi_episode": False},
        "E": {"lambda_cali_cls": 0.5, "centered_cali": True, "multi_episode": True},
    }
    if variant not in settings:
        raise ValueError("ablation.variant must be one of A, B, C, D, E")
    for key, value in settings[variant].items():
        OmegaConf.update(cfg, f"ablation.{key}", value, force_add=True)
    OmegaConf.update(cfg, "ablation.variant", variant, force_add=True)
    original.logger.info("V3 domain-generalization variant %s: %s", variant, settings[variant])


@hydra.main(
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="fed-main",
    version_base=None,
)
def main(cfg: DictConfig):
    _configure_variant(cfg)
    cfg.EditGrad.DE_type = "conv"
    original._compute_shared_editgrad = compute_shared_editgrad_v3
    install_shared_editgrad_v3.original_install = original._install_shared_editgrad
    original._install_shared_editgrad = install_shared_editgrad_v3
    original.FederatedGFM = V3FederatedGFM
    original._run_local_training = run_local_training_v3

    real_save = torch.save
    tracker = MovingAverageBestTracker(
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
        real_save(_add_v3_metadata(final), best_path)
        original.logger.info(
            "V3 best model: round=%d metric=%.4f path=%s",
            tracker.best_round,
            tracker.best_metric,
            best_path,
        )


if __name__ == "__main__":
    defaults = {
        "+ablation.variant=E": "ablation.variant=",
        "+ablation.centered_cali=true": "ablation.centered_cali=",
        "+ablation.multi_episode=true": "ablation.multi_episode=",
        "+ablation.sample_domain_responses=true": "ablation.sample_domain_responses=",
        "+ablation.editgrad_bank_size=32": "ablation.editgrad_bank_size=",
        "+ablation.editgrad_k_shot=5": "ablation.editgrad_k_shot=",
        "+ablation.editgrad_m_way=5": "ablation.editgrad_m_way=",
        "+ablation.diversity_weight=0.001": "ablation.diversity_weight=",
        "+ablation.stability_weight=0.01": "ablation.stability_weight=",
        "+ablation.projection_patience=40": "ablation.projection_patience=",
        "+ablation.temperature=0.1": "ablation.temperature=",
        "+ablation.cali_scale=0.1": "ablation.cali_scale=",
        "+ablation.lambda_cali_cls=0.5": "ablation.lambda_cali_cls=",
        "+ablation.lambda_consistency=0.1": "ablation.lambda_consistency=",
        "+ablation.lambda_rank=0.0": "ablation.lambda_rank=",
        "+ablation.rank_margin=0.01": "ablation.rank_margin=",
        "+ablation.best_window=5": "ablation.best_window=",
    }
    for default, key in defaults.items():
        if not any(key in argument for argument in sys.argv[1:]):
            sys.argv.append(default)
    main()
