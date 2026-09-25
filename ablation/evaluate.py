pass
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ablation.evaluation_core import _mean_std
from ablation.evaluator_base import V3DGEvaluator
from train import BatchedSQV
from utils.utils import seed_setting


class V4SQVEvaluator(V3DGEvaluator):
    def __init__(self, model_path, gpu_id, seed, query_chunk_size=8192):
        super().__init__(model_path, gpu_id, seed)
        if "v4_classifier" not in self.learner.model_state:
            raise ValueError("Checkpoint has no V4 SQV metadata; use a train.py checkpoint")
        self.sqv = BatchedSQV(self.learner.cfg).to(self.device)
        state = self.learner.model_state["model_state_dict"]
        sqv_state = {
            key[len("sqv."):]: value
            for key, value in state.items()
            if key.startswith("sqv.")
        }
        if not sqv_state:
            raise ValueError("Checkpoint model_state_dict has no sqv.* parameters")
        self.sqv.load_state_dict(sqv_state, strict=True)
        self.sqv.eval()
        self.query_chunk_size = int(query_chunk_size)
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")

    @staticmethod
    def _filter_classes(graph, k_shot, m_way):
        train_mask, test_mask = graph.train_mask.bool(), graph.test_mask.bool()
        candidates = graph.y[train_mask].unique().sort().values
        candidates = candidates[torch.isin(candidates, graph.y[test_mask].unique())]
        eligible, excluded = [], []
        for class_id in candidates:
            train_count = int((train_mask & (graph.y == class_id)).sum())
            test_count = int((test_mask & (graph.y == class_id)).sum())
            if train_count >= k_shot and test_count >= 1:
                eligible.append(class_id)
            else:
                excluded.append({
                    "class_id": int(class_id),
                    "train_count": train_count,
                    "test_count": test_count,
                })
        if not eligible:
            raise ValueError(f"No class supports k_shot={k_shot}")
        classes = torch.stack(eligible).to(graph.y.device)
        if m_way is not None:
            if classes.numel() < m_way:
                raise ValueError(
                    f"Only {classes.numel()} eligible classes, requested m_way={m_way}"
                )
            classes = classes[:m_way]
        return classes, excluded

    def _sqv_logits(self, representation, support, query, num_classes, k_shot):
        support_targets = torch.arange(
            num_classes, device=self.device
        ).repeat_interleave(k_shot)
        chunks = []
        with torch.no_grad():
            for start in range(0, query.numel(), self.query_chunk_size):
                selected = query[start:start + self.query_chunk_size]
                chunks.append(self.sqv(
                    representation[selected],
                    representation[support],
                    support_targets,
                    num_classes,
                ))
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _accuracy(logits, labels, query, classes):
        predictions = classes[logits.argmax(dim=-1)]
        return float((predictions == labels[query]).float().mean().cpu())

    def evaluate_sqv(self, dataset, k_shot, episodes, m_way):
        graph = self.load_graph(dataset, k_shot)
        classes, excluded = self._filter_classes(graph, k_shot, m_way)
        with torch.no_grad():
            hidden, _ = self.learner.backbone_gnn.encode(
                graph.x,
                graph.edge_index,
                graph.xe if hasattr(graph, "xe") else None,
                graph.batch,
            )
        reference_repr = self._cali_representation(hidden, self.reference)
        records = {
            name: [] for name in (
                "sqv_backbone", "sqv_cali_reference", "sqv_cali_support"
            )
        }
        paired = []
        for episode in range(episodes):
            support, query = self._episode_indices(
                graph, classes, k_shot, self.seed + episode
            )
            embedding = self._delta_embedding(graph, support)
            cali_repr = self._cali_representation(hidden, embedding)
            values = {}
            for name, representation in (
                ("sqv_backbone", hidden),
                ("sqv_cali_reference", reference_repr),
                ("sqv_cali_support", cali_repr),
            ):
                logits = self._sqv_logits(
                    representation, support, query, int(classes.numel()), k_shot
                )
                values[name] = self._accuracy(logits, graph.y, query, classes)
                records[name].append(values[name])
            paired.append({"episode": episode, "seed": self.seed + episode, **values})

        baseline = np.asarray(records["sqv_backbone"], dtype=np.float64)
        summary = {}
        for name, values in records.items():
            mean, std = _mean_std(values)
            differences = np.asarray(values, dtype=np.float64) - baseline
            delta, delta_std = _mean_std(differences)
            sem = delta_std / math.sqrt(max(1, episodes))
            summary[name] = {
                "accuracy_mean": mean,
                "accuracy_std": std,
                "delta_vs_sqv_backbone": delta,
                "delta_95ci": [delta - 1.96 * sem, delta + 1.96 * sem],
            }
        return {
            "dataset": dataset,
            "k_shot": k_shot,
            "m_way": int(classes.numel()),
            "episodes": episodes,
            "excluded_classes": excluded,
            "summary": summary,
            "paired_episodes": paired,
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", "--model", dest="model_path", required=True)
    parser.add_argument(
        "--datasets", nargs="+",
        default=["cora", "ogbn-products", "computers", "physics", "blogcatalog"],
    )
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--m-way", type=int)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-chunk-size", type=int, default=8192)
    parser.add_argument("--output", default="ablation/results/v4-sqv.json")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_setting(args.seed)
    evaluator = V4SQVEvaluator(
        args.model_path, args.gpu_id, args.seed, args.query_chunk_size
    )
    results = {
        dataset: evaluator.evaluate_sqv(
            dataset, args.k_shot, args.episodes, args.m_way
        )
        for dataset in args.datasets
    }
    macro = {
        name: float(np.mean([
            result["summary"][name]["accuracy_mean"]
            for result in results.values()
        ]))
        for name in ("sqv_backbone", "sqv_cali_support")
    }
    macro["cali_delta"] = macro["sqv_cali_support"] - macro["sqv_backbone"]
    payload = {
        "checkpoint": args.model_path,
        "inference_classifier": "support_query_value_attention",
        "macro": macro,
        "datasets": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(macro, indent=2))
    for dataset, result in results.items():
        summary = result["summary"]
        base = summary["sqv_backbone"]["accuracy_mean"]
        cali = summary["sqv_cali_support"]["accuracy_mean"]
        print(f"{dataset:14s} sqv={base:.4f} sqv_cali={cali:.4f} delta={cali-base:+.4f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
