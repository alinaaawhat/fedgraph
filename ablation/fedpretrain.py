pass

from __future__ import annotations

import copy
import logging
import os
import random
import os.path as osp
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, Iterable, Mapping

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from data_process.datahelper import (
    filter_unnecessary_attrs,
    refine_dataset,
    span_node_and_edge_idx,
)
from data_process.task_constructor import UnifiedTaskConstructor, train_task_constructor
from model.base import BackboneGNN, BackboneGNN2, FlexibleBackboneGNN
from model.editgrad import DomainEmbedder
from model.pt_model import GFM, PAMA
from utils.exp import init_exp
from utils.logging import logger, timer
from utils.utils import seed_setting


def _as_list(value) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _cfg_int(cfg: DictConfig, path: str, default: int) -> int:
    return int(OmegaConf.select(cfg, path, default=default))


def _configured_cuda(cfg: DictConfig):
    return OmegaConf.select(
        cfg, "federated.cuda", default=cfg.pretrain.gpu
    )


def _restrict_visible_cuda(cfg: DictConfig):
    pass
    configured_cuda = _configured_cuda(cfg)
    if configured_cuda is None:
        return
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA was initialized before GPU visibility was restricted. "
            "Start the process with CUDA_VISIBLE_DEVICES set."
        )
    physical_gpu = int(configured_cuda)
    if physical_gpu < 0:
        raise ValueError("federated.cuda must be a non-negative GPU index")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)


def _training_device(cfg: DictConfig) -> torch.device:
    if _configured_cuda(cfg) is None or not torch.cuda.is_available():
        return torch.device("cpu")
    
    return torch.device("cuda:0")


def create_backbone_and_frozen_copy(
    input_dim: int,
    num_classes: int,
    cfg: DictConfig,
    device: torch.device,
):
    pass
    n_editgrad_layers = int(cfg.EditGrad.get("n_layers_editgrad", 1))
    if cfg.EditGrad.get("use_flexible_backbone", True):
        backbone = FlexibleBackboneGNN(
            in_dim=input_dim,
            num_classes=num_classes,
            cfg=cfg,
            editgrad_layers=n_editgrad_layers,
        ).to(device)
        frozen_backbone = backbone.create_editgrad_copy().to(device)
    else:
        backbone = BackboneGNN2(
            in_dim=input_dim, num_classes=num_classes, cfg=cfg
        ).to(device)
        frozen_cfg = copy.deepcopy(cfg)
        frozen_cfg.EditGrad.n_layers = n_editgrad_layers
        frozen_backbone = BackboneGNN2(
            in_dim=input_dim, num_classes=num_classes, cfg=frozen_cfg
        ).to(device)
        frozen_backbone.load_state_dict(
            backbone.get_submodel_state_dict(n_editgrad_layers), strict=False
        )
    return backbone, frozen_backbone


def _aligned_features(dataset, cfg: DictConfig) -> torch.Tensor:
    pass
    unify_dim = int(cfg.unify_dim or 50)
    cache_path = osp.join(dataset.processed_dir, f"pca_{unify_dim}.pt")
    if osp.exists(cache_path):
        features = torch.load(
            cache_path, map_location="cpu", weights_only=False
        )
        if features.shape == (dataset.num_nodes, unify_dim):
            return features.float()
        logger.warning(
            "Ignoring incompatible PCA cache %s with shape %s",
            cache_path,
            tuple(features.shape),
        )

    source = dataset.data.xn.detach().cpu()
    if source.ndim != 2:
        raise ValueError(
            f"Client {dataset.ds_name} has invalid xn shape {tuple(source.shape)}"
        )
    if source.size(1) == unify_dim:
        features = source.clone().float()
    else:
        projected = PCA(n_components=unify_dim).fit_transform(source.numpy())
        features = torch.from_numpy(projected).float()
    torch.save(features, cache_path)
    return features


def _build_client_graph(
    dataset,
    dataset_name: str,
    label_offset: int,
    cfg: DictConfig,
) -> Data:
    pass
    x = _aligned_features(dataset, cfg)
    edge_index = dataset.edge_index.detach().cpu()
    labels = dataset.labels.reshape(-1).long().cpu() + label_offset
    num_nodes = int(dataset.num_nodes)
    num_edges = int(edge_index.size(1))

    if hasattr(dataset.data, "xe") and dataset.data.xe is not None:
        xe = dataset.data.xe.detach().cpu().reshape(num_edges, -1)
    else:
        xe = torch.zeros((num_edges, 1), dtype=torch.long)

    graph = Data(
        x=x,
        edge_index=edge_index,
        y=labels,
        xe=xe,
        batch=torch.zeros(num_nodes, dtype=torch.long),
        ptr=torch.tensor([0, num_nodes], dtype=torch.long),
        name_dict={dataset_name: 0},
    )
    if cfg.pretrain.use_original_mask:
        for attr in ("train_mask", "val_mask", "test_mask"):
            value = getattr(dataset, attr, None)
            if value is not None:
                setattr(graph, attr, value.detach().cpu())
    return graph


def get_pretrain_data(
    cfg: DictConfig,
) -> tuple[Dict[str, Data], list[str], int]:
    pass
    dataset_names = _as_list(cfg.pretrain.pretrain_datasets)
    pretrain_tasks = _as_list(cfg.pretrain.train_tasks)
    logger.info("Federated clients: %s", dataset_names)

    tasks: UnifiedTaskConstructor = train_task_constructor(
        data_path=cfg.dirs.data_storage,
        cfg=cfg,
        pretrain_tasks=pretrain_tasks,
    )
    datasets = OrderedDict()
    for dataset_name in dataset_names:
        data_config = cfg.data_config[dataset_name]
        dataset = tasks.get_dataset(cfg, data_config)
        dataset = refine_dataset(dataset)
        dataset = span_node_and_edge_idx(dataset)
        dataset = filter_unnecessary_attrs(dataset)
        datasets[dataset_name] = dataset

    
    label_offsets = []
    global_num_classes = 0
    for dataset in datasets.values():
        label_offsets.append(global_num_classes)
        global_num_classes += int(dataset.num_classes)

    clients = OrderedDict()
    for (dataset_name, dataset), offset in zip(datasets.items(), label_offsets):
        clients[dataset_name] = _build_client_graph(
            dataset, dataset_name, offset, cfg
        )
        logger.info(
            "Client %-20s nodes=%d edges=%d classes=%d label_offset=%d",
            dataset_name,
            dataset.num_nodes,
            dataset.edge_index.size(1),
            dataset.num_classes,
            offset,
        )
    return clients, pretrain_tasks, global_num_classes


class CachedEpisodeDataset(Dataset):
    pass

    def __init__(
        self,
        data: Data,
        k: int,
        m: int,
        n: int,
        t: int = 1,
        repetitions: int = 1,
    ):
        super().__init__()
        self.k = k
        self.m = m
        self.n = n
        self.t = t
        self.repetitions = repetitions
        self.num_graphs = int(data.ptr.numel() - 1)
        if self.num_graphs <= 0 or self.n <= 0 or self.repetitions <= 0:
            raise ValueError(
                "Episode dataset dimensions and repetitions must be positive"
            )

        
        self.graph_classes = []
        self.class_node_indices = []
        for graph_id in range(self.num_graphs):
            node_indices = (data.batch == graph_id).nonzero(
                as_tuple=False
            ).view(-1)
            labels = data.y[node_indices]
            classes = labels.unique()
            if classes.numel() == 0:
                raise ValueError(f"Graph {graph_id} contains no labeled nodes")
            self.graph_classes.append(classes)
            self.class_node_indices.append(
                [node_indices[labels == label] for label in classes]
            )

    def __len__(self):
        return self.num_graphs * self.n * self.repetitions

    def __getitem__(self, idx):
        base_idx = idx % (self.num_graphs * self.n)
        graph_id = base_idx // self.n
        classes = self.graph_classes[graph_id]
        pools = self.class_node_indices[graph_id]

        num_classes = min(self.m, classes.numel())
        picked_positions = torch.randperm(classes.numel())[:num_classes]
        picked_classes = classes[picked_positions]

        support = []
        query = []
        for position in picked_positions.tolist():
            pool = pools[position]
            support_count = min(self.k, pool.numel() // 2)
            query_count = min(self.t, pool.numel() - support_count)
            draw_count = support_count + query_count

            
            
            sampled_positions = torch.tensor(
                random.sample(range(pool.numel()), draw_count),
                dtype=torch.long,
            )
            sampled_nodes = pool[sampled_positions]
            support.append(sampled_nodes[:support_count])
            query.append(sampled_nodes[support_count:])

        return {
            "graph": graph_id,
            "classes": picked_classes,
            "support": torch.cat(support),
            "query": torch.cat(query),
        }


class GPUEpisodeLoader:
    pass

    def __init__(
        self,
        data: Data,
        k: int,
        m: int,
        n: int,
        t: int,
        num_batches: int,
        seed: int,
    ):
        if data.x.device.type != "cuda":
            raise ValueError("GPUEpisodeLoader requires the client graph on CUDA")
        self.device = data.x.device
        self.k = int(k)
        self.t = int(t)
        self.num_episodes = int(n)
        self.num_batches = int(num_batches)
        draw_count = self.k + self.t
        if min(self.k, self.t, self.num_episodes, self.num_batches) <= 0:
            raise ValueError("GPU episode dimensions must be positive")

        labels = data.y
        valid_nodes = (labels >= 0).nonzero(as_tuple=False).view(-1)
        valid_labels = labels[valid_nodes]
        classes, inverse, counts = torch.unique(
            valid_labels, sorted=True, return_inverse=True, return_counts=True
        )
        eligible = counts >= draw_count
        if not bool(eligible.any()):
            raise ValueError(
                f"No class contains the required k+t={draw_count} nodes"
            )

        
        
        
        kept_classes = classes[eligible]
        kept_counts = counts[eligible]
        pools = []
        eligible_positions = eligible.nonzero(as_tuple=False).view(-1)
        for class_position in eligible_positions:
            pool = valid_nodes[inverse == class_position]
            pools.append(pool[torch.randperm(pool.numel(), device=self.device)])
        self.flat_nodes = torch.cat(pools)
        self.classes = kept_classes
        self.counts = kept_counts
        self.starts = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=self.device),
                kept_counts.cumsum(0)[:-1],
            ]
        )
        self.m = min(int(m), int(self.classes.numel()))
        if self.m <= 0:
            raise ValueError("No classes are available for GPU episodes")
        self.draw_offsets = torch.arange(draw_count, device=self.device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            
            scores = torch.rand(
                self.num_episodes,
                self.classes.numel(),
                generator=self.generator,
                device=self.device,
            )
            class_positions = scores.topk(self.m, dim=1).indices
            selected_classes = self.classes[class_positions]
            selected_counts = self.counts[class_positions]
            selected_starts = self.starts[class_positions]

            random_offsets = (
                torch.rand(
                    self.num_episodes,
                    self.m,
                    generator=self.generator,
                    device=self.device,
                )
                * selected_counts
            ).long()
            pool_positions = (
                random_offsets.unsqueeze(-1) + self.draw_offsets
            ).remainder(selected_counts.unsqueeze(-1))
            sampled_nodes = self.flat_nodes[
                selected_starts.unsqueeze(-1) + pool_positions
            ]
            yield {
                "graph": torch.zeros(
                    self.num_episodes, dtype=torch.long, device=self.device
                ),
                "classes": selected_classes,
                "support": sampled_nodes[:, :, : self.k].reshape(
                    self.num_episodes, -1
                ),
                "query": sampled_nodes[:, :, self.k :].reshape(
                    self.num_episodes, -1
                ),
            }



class FederatedDataModule(pl.LightningDataModule):
    pass

    def __init__(self, data: Data, cfg: DictConfig):
        super().__init__()
        n_episodes = int(cfg.pretrain.n_eqisodes)
        k_shot = int(cfg.pretrain.k_shot)
        m_way = int(cfg.pretrain.m_way)
        t_query = int(cfg.pretrain.t_query)
        local_epochs = _cfg_int(cfg, "federated.local_epochs", 1)
        seed = int(cfg.pretrain.seed)

        self.train_loader = GPUEpisodeLoader(
            data=data,
            k=k_shot,
            m=m_way,
            n=n_episodes,
            t=t_query,
            num_batches=local_epochs,
            seed=seed,
        )
        self.val_loader = GPUEpisodeLoader(
            data=data,
            k=k_shot,
            m=m_way,
            n=max(1, n_episodes // 5),
            t=t_query,
            num_batches=1,
            seed=seed + 1_000_000,
        )
        self.test_loader = GPUEpisodeLoader(
            data=data,
            k=k_shot,
            m=m_way,
            n=max(1, n_episodes // 5),
            t=t_query,
            num_batches=1,
            seed=seed + 2_000_000,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class BatchedPAMA(PAMA):
    pass

    def forward(self, z_q, Z_sup, U_sup):
        input_dim = z_q.dim()
        if input_dim == 1:
            z_q = z_q.unsqueeze(0).unsqueeze(0)
            Z_sup = Z_sup.unsqueeze(0)
            U_sup = U_sup.unsqueeze(0)
        elif input_dim == 2:
            z_q = z_q.unsqueeze(0)
            Z_sup = Z_sup.unsqueeze(0)
            U_sup = U_sup.unsqueeze(0)
        elif input_dim != 3:
            raise ValueError(
                "z_q must have shape [h], [Q, h], or [B, Q, h]"
            )

        batch_size, num_queries, _ = z_q.shape
        num_support = Z_sup.size(1)
        z_q = self.ln(z_q)
        Z_sup = self.ln(Z_sup)

        Q_feat = self.W_Q(z_q).view(
            batch_size, num_queries, self.heads, self.d
        ).permute(0, 2, 1, 3)
        K_feat = self.W_K(Z_sup).view(
            batch_size, num_support, self.heads, self.d
        ).permute(0, 2, 1, 3)
        V_feat = self.W_V(Z_sup).view(
            batch_size, num_support, self.heads, self.d
        ).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(
            Q_feat, K_feat.transpose(-2, -1)
        ) / (self.d ** 0.5)
        attn_weights = self.dropout_layer(
            torch.softmax(attn_scores, dim=-1)
        )
        z_out = torch.matmul(attn_weights, V_feat)
        z_out = z_out.permute(0, 2, 1, 3).reshape(
            batch_size, num_queries, -1
        )
        z_out = self.W_O(z_out)
        z_hat = self.g(z_out)

        V_label = self.W_V(U_sup)
        V_label = V_label.view(
            batch_size, U_sup.size(1), self.heads, self.d
        ).mean(dim=2)
        logits = torch.matmul(
            z_hat, V_label.transpose(-1, -2)
        ) / 0.1

        if input_dim == 1:
            return logits[0, 0]
        if input_dim == 2:
            return logits[0]
        return logits


class FederatedGFM(GFM):
    pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        original_pama_state = self.pama.state_dict()
        self.pama = BatchedPAMA(self.cfg)
        self.pama.load_state_dict(original_pama_state)
        self.local_metrics = OrderedDict()
        self.episode_batch_size = _cfg_int(
            self.cfg, "federated.episode_batch_size", 32
        )
        if self.episode_batch_size <= 0:
            raise ValueError("federated.episode_batch_size must be positive")

    def reset_local_metrics(self):
        self.local_metrics.clear()

    def log(self, name, value, *args, **kwargs):
        
        
        if isinstance(value, torch.Tensor):
            
            
            value = value.detach()
        elif isinstance(value, (int, float)):
            value = float(value)
        else:
            return
        self.local_metrics[name] = value

    def _episode_batch_step(self, batch, training: bool):
        if self.gamma_f is None:
            self._compute_domain_embeddings()
        graph = self.comb_pretrained_graphs
        H, _ = self.forward_backbone(
            graph.x,
            graph.edge_index,
            graph.xe if hasattr(graph, "xe") else None,
            graph.batch,
        )

        if isinstance(batch, Mapping):
            idx_sup = batch["support"]
            idx_qry = batch["query"]
            classes = batch["classes"]
            gids = batch["graph"]
            batch_size, num_queries = idx_qry.shape
            num_classes = classes.size(1)

            gamma_f = self.gamma_f[gids].unsqueeze(1)
            beta_f = self.beta_f[gids].unsqueeze(1)
            z_sup = gamma_f * H[idx_sup] + beta_f
            z_qry = gamma_f * H[idx_qry] + beta_f

            gamma_l = self.gamma_l[gids].unsqueeze(1)
            beta_l = self.beta_l[gids].unsqueeze(1)
            U_sup = gamma_l * self.E_lab[classes] + beta_l
            if training:
                z_sup = F.normalize(z_sup, p=2, dim=-1)
                z_qry = F.normalize(z_qry, p=2, dim=-1)
                U_sup = F.normalize(U_sup, p=2, dim=-1)

            logits = self.pama(z_qry, z_sup, U_sup)
            queries_per_class = num_queries // num_classes
            targets = torch.arange(
                num_classes, device=self.device
            ).repeat_interleave(queries_per_class)
            targets = targets.unsqueeze(0).expand(batch_size, -1)
            loss = F.cross_entropy(
                logits.reshape(-1, num_classes),
                targets.reshape(-1),
                label_smoothing=0.1 if training else 0.0,
            )
            accuracy = (logits.argmax(dim=-1) == targets).float().mean()
            return loss, accuracy


        
        
        shape_groups = OrderedDict()
        for episode in batch:
            shape = (
                episode["query"].numel(),
                episode["support"].numel(),
                episode["classes"].numel(),
            )
            shape_groups.setdefault(shape, []).append(episode)

        episode_losses = []
        episode_accuracies = []
        for (num_queries, _, num_classes), episodes in shape_groups.items():
            for start in range(0, len(episodes), self.episode_batch_size):
                chunk = episodes[start : start + self.episode_batch_size]
                idx_sup = torch.stack(
                    [episode["support"] for episode in chunk]
                ).to(self.device, non_blocking=True)
                idx_qry = torch.stack(
                    [episode["query"] for episode in chunk]
                ).to(self.device, non_blocking=True)
                classes = torch.stack(
                    [episode["classes"] for episode in chunk]
                ).to(self.device, non_blocking=True)
                gids = torch.tensor(
                    [episode["graph"] for episode in chunk],
                    device=self.device,
                    dtype=torch.long,
                )

                gamma_f = self.gamma_f[gids].unsqueeze(1)
                beta_f = self.beta_f[gids].unsqueeze(1)
                z_sup = gamma_f * H[idx_sup] + beta_f
                z_qry = gamma_f * H[idx_qry] + beta_f

                gamma_l = self.gamma_l[gids].unsqueeze(1)
                beta_l = self.beta_l[gids].unsqueeze(1)
                U_sup = gamma_l * self.E_lab[classes] + beta_l

                if training:
                    z_sup = F.normalize(z_sup, p=2, dim=-1)
                    z_qry = F.normalize(z_qry, p=2, dim=-1)
                    U_sup = F.normalize(U_sup, p=2, dim=-1)

                
                
                logits = self.pama(z_qry, z_sup, U_sup)
                targets = torch.arange(
                    num_queries, device=self.device
                ).remainder(num_classes)
                targets = targets.unsqueeze(0).expand(len(chunk), -1)

                losses = F.cross_entropy(
                    logits.reshape(-1, num_classes),
                    targets.reshape(-1),
                    label_smoothing=0.1 if training else 0.0,
                    reduction="none",
                ).view(len(chunk), num_queries)
                accuracies = (
                    logits.argmax(dim=-1) == targets
                ).float()
                episode_losses.append(losses.mean(dim=1))
                episode_accuracies.append(accuracies.mean(dim=1))

        return (
            torch.cat(episode_losses).mean(),
            torch.cat(episode_accuracies).mean(),
        )

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._episode_batch_step(batch, training=True)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._episode_batch_step(batch, training=False)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

def _client_cfg(cfg: DictConfig, client_name: str) -> DictConfig:
    local_cfg = copy.deepcopy(cfg)
    local_cfg.pretrain.pretrain_datasets = [client_name]
    run_name = osp.basename(str(cfg.dirs.output).rstrip(osp.sep))
    
    
    local_cfg.dirs.editgrad_storage = osp.join(
        cfg.dirs.editgrad_storage,
        "federated_clients",
        run_name,
    )
    return local_cfg


def _shared_editgrad_cfg(cfg: DictConfig) -> DictConfig:
    pass
    shared_cfg = copy.deepcopy(cfg)
    run_name = osp.basename(str(cfg.dirs.output).rstrip(osp.sep))
    shared_cfg.dirs.editgrad_storage = osp.join(
        cfg.dirs.editgrad_storage,
        "federated_shared",
        run_name,
    )
    return shared_cfg


def _compute_shared_editgrad(
    cfg: DictConfig,
    clients: Mapping[str, Data],
    frozen_backbone: torch.nn.Module,
    device: torch.device,
):
    pass




    shared_cfg = _shared_editgrad_cfg(cfg)
    shared_embedder = DomainEmbedder(
        frozen_backboneGNN=copy.deepcopy(frozen_backbone).to(device),
        cfg=shared_cfg,
    ).to(device)
    extractor = shared_embedder.dm_extractor

    deltas = [
        extractor._probe_grad4domain(graph, None, 0)
        for graph in clients.values()
    ]
    if shared_cfg.EditGrad.DE_type == "pca":
        stacked_deltas = torch.stack(deltas, dim=0).to(device)
        basis, embeddings = extractor._fit_pca(stacked_deltas)
        extractor._B = basis
        extractor._e = embeddings
        extractor._delta_matrices = None
    elif shared_cfg.EditGrad.DE_type == "conv":
        feature_dim = int(deltas[0].shape[0])
        max_width = max(int(delta.shape[1]) for delta in deltas)
        padding_strategy = shared_cfg.EditGrad.DE.get(
            "padding_strategy", "zero"
        )
        padding_noise_std = float(
            shared_cfg.EditGrad.DE.get("padding_noise_std", 0.01)
        )
        padded_deltas = []
        for delta in deltas:
            missing = max_width - int(delta.shape[1])
            if missing <= 0:
                padded = delta
            elif padding_strategy == "noise":
                padding = torch.randn(
                    feature_dim, missing, device=delta.device
                ) * padding_noise_std
                padded = torch.cat((delta, padding), dim=1)
            elif padding_strategy == "repeat_last":
                padding = delta[:, -1:].repeat(1, missing)
                padded = torch.cat((delta, padding), dim=1)
            else:
                padding = torch.zeros(
                    feature_dim, missing, device=delta.device
                )
                padded = torch.cat((delta, padding), dim=1)
            padded_deltas.append(padded)

        extractor._delta_matrices = deltas
        extractor._padded_delta_matrices = padded_deltas
        extractor._original_shapes = [delta.shape for delta in deltas]
        extractor._d_c_max = max_width
        extractor._train_projection(padded_deltas)
        embeddings = []
        for delta in padded_deltas:
            with torch.no_grad():
                embedding = extractor.projection(delta.to(device))
                if shared_cfg.EditGrad.l2_normalize:
                    embedding = F.normalize(embedding, p=2, dim=-1)
                embeddings.append(embedding)
        extractor._e = torch.stack(embeddings, dim=0)
    else:
        raise ValueError(
            f"Unsupported editgrad type {shared_cfg.EditGrad.DE_type!r}"
        )

    extractor._save_cache()
    logger.info(
        "Computed one shared editgrad space from %d client domains",
        len(clients),
    )
    return shared_embedder


def _install_shared_editgrad(
    model: "FederatedGFM",
    shared_embedder: DomainEmbedder,
    client_index: int,
):
    pass
    source = shared_embedder.dm_extractor
    target = model.de.dm_extractor
    target._theta0 = {
        name: tensor.detach().cpu().clone()
        for name, tensor in source._theta0.items()
    }
    target.frozen_backbone.load_state_dict(target._theta0, strict=False)
    target._e = source._e[client_index : client_index + 1].detach().clone()
    target._cached = True

    if model.cfg.EditGrad.DE_type == "pca":
        target._B = source._B.detach().clone()
    else:
        target.projection.load_state_dict(source.projection.state_dict())
        target.projection.eval()
        target._delta_matrices = [
            source._delta_matrices[client_index].detach().cpu().clone()
        ]
        target._d_c_max = int(source._d_c_max)

    model.domain_embeddings = None
    model.gamma_f = None
    model.beta_f = None
    model.gamma_l = None
    model.beta_l = None



def _build_model(
    cfg: DictConfig,
    graph: Data,
    num_classes: int,
    device: torch.device,
) -> FederatedGFM:
    input_dim = int(graph.x.size(-1))
    if cfg.EditGrad.n_layers == 1 and cfg.EditGrad.readout_proj:
        backbone = BackboneGNN(
            in_dim=input_dim, num_classes=num_classes, cfg=cfg
        ).to(device)
        frozen_backbone = copy.deepcopy(backbone)
        frozen_backbone.n_layers = 1
        frozen_backbone.readout_proj = False
    else:
        backbone, frozen_backbone = create_backbone_and_frozen_copy(
            input_dim, num_classes, cfg, device
        )
    domain_embedder = DomainEmbedder(
        frozen_backboneGNN=frozen_backbone, cfg=cfg
    ).to(device)
    return FederatedGFM(
        cfg=cfg,
        L_max=num_classes,
        comb_pretrained_graphs=graph,
        backboneGNN=backbone,
        domain_embedder=domain_embedder,
    )


def _clone_state_dict(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach().clone())
        for name, tensor in model.state_dict().items()
    )


def _state_to_cpu(state: Mapping[str, torch.Tensor]):
    return OrderedDict(
        (name, tensor.detach().cpu().clone()) for name, tensor in state.items()
    )


def fedavg(
    client_states: Iterable[Mapping[str, torch.Tensor]],
    client_weights: Iterable[int],
) -> OrderedDict[str, torch.Tensor]:
    pass
    states = list(client_states)
    weights = [float(weight) for weight in client_weights]
    if not states or len(states) != len(weights):
        raise ValueError("FedAvg requires one positive weight per client state")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("FedAvg client weights must sum to a positive value")

    reference_keys = list(states[0].keys())
    for state in states[1:]:
        if list(state.keys()) != reference_keys:
            raise ValueError("Client model state dictionaries do not match")

    averaged = OrderedDict()
    for key in reference_keys:
        reference = states[0][key]
        if reference.is_floating_point() or reference.is_complex():
            value = torch.zeros_like(reference)
            for state, weight in zip(states, weights):
                value.add_(state[key], alpha=weight / total_weight)
            averaged[key] = value
        else:
            averaged[key] = reference.cpu().clone()
    return averaged


def _extract_prefixed_state(
    state: Mapping[str, torch.Tensor], prefix: str
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name[len(prefix):], tensor)
        for name, tensor in state.items()
        if name.startswith(prefix)
    )


def _metric_value(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return value

def _aggregate_server_metrics(client_metrics, client_weights):
    pass
    weights = dict(zip(client_metrics.keys(), map(float, client_weights)))
    aliases = OrderedDict(
        [
            ("train_loss", ("train_loss_epoch", "train_loss", "train_loss_step")),
            ("train_acc", ("train_acc_epoch", "train_acc", "train_acc_step")),
            ("val_loss", ("val_loss",)),
            ("val_acc", ("val_acc",)),
        ]
    )
    server_metrics = OrderedDict()
    for output_name, candidates in aliases.items():
        for candidate in candidates:
            observed = [
                (metrics[candidate], weights[client_name])
                for client_name, metrics in client_metrics.items()
                if candidate in metrics
                and isinstance(metrics[candidate], (int, float))
            ]
            if observed:
                total_weight = sum(weight for _, weight in observed)
                server_metrics[output_name] = sum(
                    value * weight for value, weight in observed
                ) / total_weight
                break
    return server_metrics




def _model_optimizer(model: FederatedGFM):
    configured = model.configure_optimizers()
    if isinstance(configured, torch.optim.Optimizer):
        return configured
    if isinstance(configured, Mapping) and "optimizer" in configured:
        return configured["optimizer"]
    raise TypeError("GFM.configure_optimizers() did not return an optimizer")


def _run_local_training(
    model: FederatedGFM,
    optimizer: torch.optim.Optimizer,
    train_loader,
    val_loader,
    local_epochs: int,
):
    pass
    model.reset_local_metrics()
    model.train()
    completed_epochs = 0
    for batch_index, batch in enumerate(train_loader):
        if completed_epochs >= local_epochs:
            break
        model.on_train_epoch_start()
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_step(batch, batch_index)
        loss.backward()
        optimizer.step()
        completed_epochs += 1
    if completed_epochs != local_epochs:
        raise RuntimeError(
            f"Expected {local_epochs} local batches, got {completed_epochs}"
        )

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            model.validation_step(batch, batch_index)
    return {
        name: _metric_value(value)
        for name, value in model.local_metrics.items()
    }



@timer()
@hydra.main(config_path=f"{root}/configs", config_name="fed-main", version_base=None)
def main(cfg: DictConfig):
    _restrict_visible_cuda(cfg)
    cfg = init_exp(cfg)
    seed_setting(int(cfg.pretrain.seed))
    device = _training_device(cfg)
    clients, pretrain_tasks, global_num_classes = get_pretrain_data(cfg)
    if not clients:
        raise ValueError("No federated clients were created")

    rounds = _cfg_int(
        cfg, "federated.rounds", int(cfg.pretrain.pretrain_epochs)
    )
    local_epochs = _cfg_int(cfg, "federated.local_epochs", 1)
    if rounds <= 0 or local_epochs <= 0:
        raise ValueError("federated.rounds and local_epochs must be positive")

    clients = OrderedDict(
        (name, graph.to(device)) for name, graph in clients.items()
    )
    first_name, first_graph = next(iter(clients.items()))
    client_models = OrderedDict()
    client_optimizers = OrderedDict()
    client_data_modules = OrderedDict()
    train_loaders = OrderedDict()
    val_loaders = OrderedDict()
    global_state = None

    for client_name, client_graph in clients.items():
        local_cfg = _client_cfg(cfg, client_name)
        local_model = _build_model(
            local_cfg, client_graph, global_num_classes, device
        ).to(device)
        if global_state is None:
            global_state = _clone_state_dict(local_model)
        else:
            local_model.load_state_dict(global_state, strict=True)
        client_models[client_name] = local_model

    shared_embedder = _compute_shared_editgrad(
        cfg=cfg,
        clients=clients,
        frozen_backbone=client_models[first_name].frozen_backbone,
        device=device,
    )
    for client_index, (client_name, client_graph) in enumerate(clients.items()):
        local_model = client_models[client_name]
        _install_shared_editgrad(
            model=local_model,
            shared_embedder=shared_embedder,
            client_index=client_index,
        )

        
        
        with open(os.devnull, "w") as client_output:
            with redirect_stdout(client_output), redirect_stderr(client_output):
                local_model.setup("fit")

        data_module = FederatedDataModule(client_graph, local_model.cfg)
        client_optimizers[client_name] = _model_optimizer(local_model)
        client_data_modules[client_name] = data_module
        train_loaders[client_name] = data_module.train_dataloader()
        val_loaders[client_name] = data_module.val_dataloader()

    
    
    global_state = _clone_state_dict(client_models[first_name])
    for local_model in client_models.values():
        local_model.load_state_dict(global_state, strict=True)

    if global_state is None:
        raise RuntimeError("Failed to initialize the global model")

    checkpoint_dir = osp.join(cfg.dirs.checkpoint_dir, "federated")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.dirs.output, exist_ok=True)
    history = []

    logger.info(
        "Starting FedAvg: rounds=%d local_epochs=%d clients=%d device=%s",
        rounds,
        local_epochs,
        len(clients),
        device,
    )
    for round_index in range(1, rounds + 1):
        local_states = []
        local_weights = []
        round_metrics = {}

        for client_name, client_graph in clients.items():
            local_model = client_models[client_name]
            local_model.load_state_dict(global_state, strict=True)
            round_metrics[client_name] = _run_local_training(
                model=local_model,
                optimizer=client_optimizers[client_name],
                train_loader=train_loaders[client_name],
                val_loader=val_loaders[client_name],
                local_epochs=local_epochs,
            )
            local_states.append(_clone_state_dict(local_model))
            
            
            local_weights.append(1)

        global_state = fedavg(local_states, local_weights)
        server_metrics = _aggregate_server_metrics(
            round_metrics, local_weights
        )
        metric_text = " ".join(
            f"{name}={value:.4f}"
            for name, value in server_metrics.items()
        )
        logger.info(
            "Server round %d/%d | %s",
            round_index, rounds, metric_text or "model aggregated",
        )
        history.append({"round": round_index, "server": server_metrics})
        torch.save(
            {
                "round": round_index,
                "global_state_dict": _state_to_cpu(global_state),
                "client_weights": dict(zip(clients.keys(), local_weights)),
                "server_metrics": server_metrics,
            },
            osp.join(checkpoint_dir, "last.pt"),
        )

    final_model_path = osp.join(
        cfg.dirs.output, "final_federated_gfm_model.pt"
    )
    client_info = {
        name: {
            "num_nodes": int(graph.num_nodes),
            "num_edges": int(graph.edge_index.size(1)),
            "num_features": int(graph.x.size(1)),
            "labels": [int(v) for v in graph.y.unique().sort().values],
        }
        for name, graph in clients.items()
    }
    node_counts = [int(graph.num_nodes) for graph in clients.values()]
    node_offsets = [0]
    for count in node_counts:
        node_offsets.append(node_offsets[-1] + count)
    graph_summary = {
        "num_nodes": sum(node_counts),
        "num_features": int(first_graph.x.size(1)),
        "num_graphs": len(clients),
        "ptr": torch.tensor(node_offsets, dtype=torch.long),
        "name_dict": {name: index for index, name in enumerate(clients)},
    }
    final_state = _state_to_cpu(global_state)
    shared_extractor = shared_embedder.dm_extractor
    model_state = {
        "federated": True,
        "algorithm": "FedAvg",
        "rounds": rounds,
        "local_epochs": local_epochs,
        "model_state_dict": final_state,
        "backbone_state_dict": _extract_prefixed_state(final_state, "GNNEnc."),
        "frozen_backbone_state_dict": _extract_prefixed_state(
            final_state, "de.frozen_backboneGNN."
        ),
        "domain_embedder_B": (
            shared_extractor._B.detach().cpu()
            if shared_extractor._B is not None
            else None
        ),
        "domain_embedder_theta0": {
            name: tensor.detach().cpu().clone()
            for name, tensor in shared_extractor._theta0.items()
        },
        "domain_embedder_e": shared_extractor._e.detach().cpu(),
        "config": cfg,
        "L_max": global_num_classes,
        "pretrain_datasets": list(clients.keys()),
        "pretrain_tasks": pretrain_tasks,
        "federated_clients_info": client_info,
        "combined_graphs_info": graph_summary,
        "history": history,
    }
    if cfg.EditGrad.DE_type == "conv":
        model_state["domain_embedder_projection_state"] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in shared_extractor.projection.state_dict().items()
        }
        model_state["domain_embedder_delta_matrices"] = [
            delta.detach().cpu().clone()
            for delta in shared_extractor._delta_matrices
        ]
        model_state["domain_embedder_d_c_max"] = int(
            shared_extractor._d_c_max
        )
    torch.save(model_state, final_model_path)
    logger.info("Federated model saved to: %s", final_model_path)
    return global_state, final_model_path


if __name__ == "__main__":
    main()
