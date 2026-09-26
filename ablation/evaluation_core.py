pass






from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import EvaluationPrototypeLearner
from model.Cali import DomainCali
from utils.utils import seed_setting


DEFAULT_MODES = (
    "backbone",
    "cali_source_mean",
    "cali_zero_delta",
    "cali_shuffled_embedding",
    "cali_support_delta",
    "cali_full_labels_leaky",
    "random_cali_support_delta",
)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    ddof = 1 if array.size > 1 else 0
    return float(array.mean()), float(array.std(ddof=ddof))


class ComponentAblator:
    def __init__(self, model_path: str, gpu_id: int, seed: int):
        args = SimpleNamespace(model_path=model_path, gpu_id=gpu_id, k_shot=5)
        self.learner = EvaluationPrototypeLearner(args)
        self.device = self.learner.device
        self.seed = seed
        self.checkpoint_cali = self.learner.domain_embedder.dm_cali.eval()
        self.random_cali = copy.deepcopy(self.checkpoint_cali)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 91_337)
            self.random_cali.apply(self._reset_module)
        self.random_cali.to(self.device).eval()

    @staticmethod
    def _reset_module(module):
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()

    def load_graph(self, dataset: str, k_shot: int):
        self.learner.args.k_shot = k_shot
        return self.learner.load_downstream_graph(dataset).to(self.device)

    def _delta_embedding(
        self,
        graph,
        label_indices: torch.Tensor | None,
        zero_delta: bool = False,
        selected_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass





        extractor = self.learner.domain_embedder.dm_extractor
        model = self.learner.frozen_backbone
        model.load_state_dict(extractor._theta0, strict=False)
        model.to(self.device).eval()
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits, _ = model(graph)
            if label_indices is None:
                selected_logits = logits
                selected_labels = graph.y
            else:
                selected_logits = logits[label_indices]
                selected_labels = (
                    graph.y[label_indices]
                    if selected_targets is None
                    else selected_targets.to(
                        device=selected_logits.device, dtype=torch.long
                    )
                )
            if selected_labels.numel() == 0:
                raise ValueError("Cannot compute a supervised delta without labels")
            if selected_labels.numel() != selected_logits.size(0):
                raise ValueError(
                    "selected_targets must contain exactly one target per selected node"
                )
            target_min = int(selected_labels.min().detach().cpu())
            target_max = int(selected_labels.max().detach().cpu())
            if target_min < 0 or target_max >= selected_logits.size(-1):
                raise ValueError(
                    "Probe targets must be in [0, "
                    f"{selected_logits.size(-1) - 1}], found "
                    f"[{target_min}, {target_max}]"
                )
            loss = F.cross_entropy(selected_logits, selected_labels)
            loss.backward()

        gradient = next(
            (
                parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
                if "weight" in name and parameter.grad is not None
            ),
            None,
        )
        if gradient is None:
            raise RuntimeError("No probe gradient was produced")
        if gradient.shape[0] != graph.x.shape[1]:
            gradient = gradient.T
        delta = -float(self.learner.cfg.EditGrad.probe_lr) * gradient
        if zero_delta:
            delta.zero_()

        width = int(extractor._d_c_max)
        if delta.shape[1] < width:
            delta = F.pad(delta, (0, width - delta.shape[1]))
        elif delta.shape[1] > width:
            delta = delta[:, :width]

        with torch.no_grad():
            embedding = extractor.projection(delta)
            if bool(self.learner.cfg.EditGrad.l2_normalize):
                embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding.detach()

    def _source_mean_embedding(self) -> torch.Tensor:
        embeddings = self.learner.model_state.get("domain_embedder_e")
        if embeddings is None:
            raise ValueError("Checkpoint has no source-domain embeddings")
        embedding = embeddings.to(self.device).mean(dim=0)
        return F.normalize(embedding, p=2, dim=-1)

    @staticmethod
    def _eligible_classes(graph) -> torch.Tensor:
        train = graph.train_mask.bool()
        test = graph.test_mask.bool()
        if train.ndim != 1 or test.ndim != 1:
            raise ValueError("Expected one-dimensional train/test masks")
        train_classes = graph.y[train].unique()
        test_classes = graph.y[test].unique()
        return train_classes[torch.isin(train_classes, test_classes)].sort().values

    def _episode_indices(self, graph, classes, k_shot: int, seed: int):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        support = []
        for class_id in classes:
            candidates = (
                graph.train_mask.bool() & (graph.y == class_id)
            ).nonzero(as_tuple=False).view(-1)
            if candidates.numel() < k_shot:
                raise ValueError(
                    f"Class {int(class_id)} has {candidates.numel()} train nodes, "
                    f"fewer than k_shot={k_shot}"
                )
            order = torch.randperm(candidates.numel(), generator=generator)
            support.append(candidates[order[:k_shot].to(candidates.device)])
        support_indices = torch.cat(support)
        query_mask = graph.test_mask.bool() & torch.isin(graph.y, classes)
        query_indices = query_mask.nonzero(as_tuple=False).view(-1)
        return support_indices, query_indices

    @staticmethod
    def _prototype_accuracy(z, labels, support, query, classes) -> float:
        z = F.normalize(z, p=2, dim=-1)
        prototypes = []
        for class_id in classes:
            prototype = z[support][labels[support] == class_id].mean(dim=0)
            prototypes.append(F.normalize(prototype, p=2, dim=-1))
        prototypes = torch.stack(prototypes)
        scores = F.normalize(z[query], p=2, dim=-1) @ prototypes.T
        predictions = classes[scores.argmax(dim=-1)]
        return float((predictions == labels[query]).float().mean().cpu())

    def _apply_cali(self, hidden, embedding, cali):
        with torch.no_grad():
            gamma, beta, _, _ = cali(embedding.unsqueeze(0))
        return gamma.squeeze(0) * hidden + beta.squeeze(0)

    def evaluate(
        self,
        dataset: str,
        k_shot: int,
        episodes: int,
        m_way: int | None,
        modes: tuple[str, ...],
    ) -> dict:
        unknown = sorted(set(modes) - set(DEFAULT_MODES))
        if unknown:
            raise ValueError(f"Unknown modes: {unknown}; valid modes={DEFAULT_MODES}")
        graph = self.load_graph(dataset, k_shot)
        classes = self._eligible_classes(graph)
        if m_way is not None:
            if classes.numel() < m_way:
                raise ValueError(
                    f"{dataset} has only {classes.numel()} eligible classes, "
                    f"but m_way={m_way}"
                )
            classes = classes[:m_way]

        with torch.no_grad():
            hidden, _ = self.learner.backbone_gnn.encode(
                graph.x,
                graph.edge_index,
                graph.xe if hasattr(graph, "xe") else None,
                graph.batch,
            )
        source_mean = self._source_mean_embedding()
        zero_embedding = self._delta_embedding(
            graph, graph.train_mask.nonzero(as_tuple=False).view(-1), zero_delta=True
        )
        full_embedding = None
        if "cali_full_labels_leaky" in modes:
            full_embedding = self._delta_embedding(graph, None)

        per_mode = {mode: [] for mode in modes}
        episode_records = []
        for episode in range(episodes):
            episode_seed = self.seed + episode
            support, query = self._episode_indices(
                graph, classes, k_shot, episode_seed
            )
            support_embedding = None
            if any("support_delta" in mode for mode in modes):
                support_embedding = self._delta_embedding(graph, support)

            record = {"episode": episode, "seed": episode_seed}
            for mode in modes:
                if mode == "backbone":
                    representation = hidden
                elif mode == "cali_source_mean":
                    representation = self._apply_cali(
                        hidden, source_mean, self.checkpoint_cali
                    )
                elif mode == "cali_zero_delta":
                    representation = self._apply_cali(
                        hidden, zero_embedding, self.checkpoint_cali
                    )
                elif mode == "cali_shuffled_embedding":
                    generator = torch.Generator(device="cpu").manual_seed(
                        episode_seed + 700_001
                    )
                    order = torch.randperm(
                        support_embedding.numel(), generator=generator
                    ).to(self.device)
                    representation = self._apply_cali(
                        hidden, support_embedding[order], self.checkpoint_cali
                    )
                elif mode == "cali_support_delta":
                    representation = self._apply_cali(
                        hidden, support_embedding, self.checkpoint_cali
                    )
                elif mode == "cali_full_labels_leaky":
                    representation = self._apply_cali(
                        hidden, full_embedding, self.checkpoint_cali
                    )
                elif mode == "random_cali_support_delta":
                    representation = self._apply_cali(
                        hidden, support_embedding, self.random_cali
                    )
                accuracy = self._prototype_accuracy(
                    representation, graph.y, support, query, classes
                )
                per_mode[mode].append(accuracy)
                record[mode] = accuracy
            episode_records.append(record)

        baseline = np.asarray(per_mode["backbone"], dtype=np.float64)
        summary = {}
        for mode, values in per_mode.items():
            mean, std = _mean_std(values)
            differences = np.asarray(values, dtype=np.float64) - baseline
            delta_mean, delta_std = _mean_std(differences)
            sem = delta_std / math.sqrt(max(1, episodes))
            summary[mode] = {
                "accuracy_mean": mean,
                "accuracy_std": std,
                "delta_vs_backbone": delta_mean,
                "delta_95ci": [delta_mean - 1.96 * sem, delta_mean + 1.96 * sem],
                "degrades_backbone": bool(delta_mean < 0),
            }
        return {
            "dataset": dataset,
            "k_shot": k_shot,
            "m_way": int(classes.numel()),
            "episodes": episodes,
            "protocol": (
                "transductive graph/features; support-only labels except the "
                "explicit cali_full_labels_leaky diagnostic"
            ),
            "summary": summary,
            "paired_episodes": episode_records,
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--datasets", nargs="+", default=["cora"])
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--m-way", type=int)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=DEFAULT_MODES)
    parser.add_argument("--output", default="ablution/results/component_ablation.json")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_setting(args.seed)
    evaluator = ComponentAblator(args.model_path, args.gpu_id, args.seed)
    results = {
        dataset: evaluator.evaluate(
            dataset,
            args.k_shot,
            args.episodes,
            args.m_way,
            tuple(args.modes),
        )
        for dataset in args.datasets
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for dataset, result in results.items():
        print(f"\n[{dataset}] paired cosine-prototype ablation")
        for mode, metrics in result["summary"].items():
            print(
                f"{mode:32s} acc={metrics['accuracy_mean']:.4f} "
                f"delta={metrics['delta_vs_backbone']:+.4f}"
            )
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()

