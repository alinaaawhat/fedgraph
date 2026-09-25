pass




from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from ablation.evaluation_core import ComponentAblator, _mean_std
from utils.utils import seed_setting


class V3DGEvaluator(ComponentAblator):
    def __init__(self, model_path, gpu_id, seed):
        super().__init__(model_path, gpu_id, seed)
        metadata = self.learner.cfg["_ds_meta_data"]
        if "blogcatalog" not in metadata:
            OmegaConf.update(
                self.learner.cfg,
                "_ds_meta_data.blogcatalog",
                "pyg, AttributedGraphDataset.BlogCatalog",
                force_add=True,
            )
        v3 = self.learner.model_state.get("v3_domain_generalization", {})
        centers = v3.get("centers")
        reference = v3.get("reference")
        if centers is None:
            centers = self.learner.model_state.get("domain_embedder_e")
        if centers is None:
            raise ValueError("Checkpoint has no source domain centers")
        self.source_centers = centers.to(self.device)
        self.reference = (
            reference.to(self.device)
            if reference is not None
            else self.source_centers.mean(dim=0)
        )
        ablation = self.learner.cfg.get("ablation", {})
        self.centered_cali = bool(ablation.get("centered_cali", True))
        self.cali_scale = float(ablation.get("cali_scale", 0.1))
        self.temperature = float(ablation.get("temperature", 0.1))

    def _cali_residual(self, embedding):
        with torch.no_grad():
            gamma, beta, _, _ = self.checkpoint_cali(embedding.unsqueeze(0))
            if self.centered_cali:
                ref_gamma, ref_beta, _, _ = self.checkpoint_cali(
                    self.reference.unsqueeze(0)
                )
                gamma = gamma - ref_gamma
                beta = beta - ref_beta
            else:
                gamma = gamma - 1.0
            delta_gamma = self.cali_scale * torch.tanh(gamma.squeeze(0))
            delta_beta = self.cali_scale * torch.tanh(beta.squeeze(0))
        return delta_gamma, delta_beta

    def _cali_representation(self, hidden, embedding):
        delta_gamma, delta_beta = self._cali_residual(embedding)
        return hidden + delta_gamma * hidden + delta_beta

    def _logits(self, representation, labels, support, query, classes):
        representation = F.normalize(representation, p=2, dim=-1)
        prototypes = []
        for class_id in classes:
            prototype = representation[support][labels[support] == class_id].mean(0)
            prototypes.append(F.normalize(prototype, p=2, dim=-1))
        prototypes = torch.stack(prototypes)
        return representation[query] @ prototypes.T / self.temperature

    def _accuracy(self, logits, query_labels, classes):
        predictions = classes[logits.argmax(dim=-1)]
        return float((predictions == query_labels).float().mean().cpu())

    def _loo_loss(self, representation, labels, support, classes):
        support_labels = labels[support]
        if min(int((support_labels == label).sum()) for label in classes) < 2:
            return None
        representation = F.normalize(representation, p=2, dim=-1)
        losses = []
        for index, node in enumerate(support):
            prototypes = []
            for class_id in classes:
                mask = support_labels == class_id
                if bool(support_labels[index] == class_id):
                    mask = mask.clone()
                    mask[index] = False
                prototype = representation[support[mask]].mean(0)
                prototypes.append(F.normalize(prototype, p=2, dim=-1))
            logits = representation[node] @ torch.stack(prototypes).T
            target = (classes == support_labels[index]).nonzero(
                as_tuple=False
            ).view(-1)
            losses.append(
                F.cross_entropy(logits.unsqueeze(0) / self.temperature, target)
            )
        return float(torch.stack(losses).mean().cpu())

    def evaluate_dataset(self, dataset, k_shot, episodes, m_way, gate_bias, gate_temperature):
        graph = self.load_graph(dataset, k_shot)
        candidate_classes = self._eligible_classes(graph)
        train_mask = graph.train_mask.bool()
        test_mask = graph.test_mask.bool()
        eligible, excluded = [], []
        for class_id in candidate_classes:
            train_count = int((train_mask & (graph.y == class_id)).sum())
            test_count = int((test_mask & (graph.y == class_id)).sum())
            record = {
                "class_id": int(class_id),
                "train_count": train_count,
                "test_count": test_count,
            }
            if train_count >= k_shot and test_count >= 1:
                eligible.append(class_id)
            else:
                excluded.append(record)
        if not eligible:
            raise ValueError(
                f"{dataset} has no class with at least k_shot={k_shot} "
                "training nodes and one test node"
            )
        classes = torch.stack(eligible).to(graph.y.device)
        if m_way is not None:
            if classes.numel() < m_way:
                raise ValueError(
                    f"{dataset} has only {classes.numel()} k-shot eligible "
                    f"classes after filtering, requested m_way={m_way}"
                )
            classes = classes[:m_way]
        with torch.no_grad():
            hidden, _ = self.learner.backbone_gnn.encode(
                graph.x,
                graph.edge_index,
                graph.xe if hasattr(graph, "xe") else None,
                graph.batch,
            )
        reference_repr = self._cali_representation(hidden, self.reference)
        zero_embedding = self._delta_embedding(
            graph,
            graph.train_mask.nonzero(as_tuple=False).view(-1),
            zero_delta=True,
        )
        zero_repr = self._cali_representation(hidden, zero_embedding)

        records = {
            name: [] for name in (
                "backbone", "cali_reference", "cali_zero",
                "cali_source_control", "cali_support", "support_gated",
                "gate", "query_cali_gain",
            )
        }
        paired = []
        for episode in range(episodes):
            support, query = self._episode_indices(
                graph, classes, k_shot, self.seed + episode
            )
            support_embedding = self._delta_embedding(graph, support)
            support_repr = self._cali_representation(hidden, support_embedding)
            source_control = self.source_centers[episode % self.source_centers.size(0)]
            source_repr = self._cali_representation(hidden, source_control)

            base_logits = self._logits(hidden, graph.y, support, query, classes)
            ref_logits = self._logits(reference_repr, graph.y, support, query, classes)
            zero_logits = self._logits(zero_repr, graph.y, support, query, classes)
            source_logits = self._logits(source_repr, graph.y, support, query, classes)
            cali_logits = self._logits(support_repr, graph.y, support, query, classes)
            base_acc = self._accuracy(base_logits, graph.y[query], classes)
            cali_acc = self._accuracy(cali_logits, graph.y[query], classes)

            base_loo = self._loo_loss(hidden, graph.y, support, classes)
            cali_loo = self._loo_loss(support_repr, graph.y, support, classes)
            if base_loo is None or cali_loo is None:
                gate = 0.0
            else:
                gate = float(torch.sigmoid(torch.tensor(
                    (base_loo - cali_loo - gate_bias) / gate_temperature
                )))
            probabilities = (
                (1.0 - gate) * F.softmax(base_logits, dim=-1)
                + gate * F.softmax(cali_logits, dim=-1)
            )
            gated_acc = self._accuracy(
                probabilities, graph.y[query], classes
            )
            values = {
                "backbone": base_acc,
                "cali_reference": self._accuracy(ref_logits, graph.y[query], classes),
                "cali_zero": self._accuracy(zero_logits, graph.y[query], classes),
                "cali_source_control": self._accuracy(source_logits, graph.y[query], classes),
                "cali_support": cali_acc,
                "support_gated": gated_acc,
                "gate": gate,
                "query_cali_gain": cali_acc - base_acc,
            }
            for name, value in values.items():
                records[name].append(value)
            paired.append({"episode": episode, "seed": self.seed + episode, **values})

        summary = {}
        baseline = np.asarray(records["backbone"])
        for name in (
            "backbone", "cali_reference", "cali_zero",
            "cali_source_control", "cali_support", "support_gated",
        ):
            mean, std = _mean_std(records[name])
            differences = np.asarray(records[name]) - baseline
            delta, delta_std = _mean_std(differences)
            sem = delta_std / math.sqrt(max(1, episodes))
            summary[name] = {
                "accuracy_mean": mean,
                "accuracy_std": std,
                "delta_vs_backbone": delta,
                "delta_95ci": [delta - 1.96 * sem, delta + 1.96 * sem],
            }
        gates = np.asarray(records["gate"], dtype=np.float64)
        gains = np.asarray(records["query_cali_gain"], dtype=np.float64)
        correlation = None
        if gates.std() > 0 and gains.std() > 0:
            correlation = float(np.corrcoef(gates, gains)[0, 1])
        summary["domain_specific_gain"] = {
            "support_minus_reference": float(
                np.mean(records["cali_support"]) - np.mean(records["cali_reference"])
            ),
            "support_minus_source_control": float(
                np.mean(records["cali_support"]) - np.mean(records["cali_source_control"])
            ),
        }
        summary["gate_diagnostics"] = {
            "activation_mean": float(gates.mean()),
            "activation_std": float(gates.std(ddof=1 if episodes > 1 else 0)),
            "support_gate_query_gain_correlation": correlation,
            "negative_transfer_count_fixed": int((gains < 0).sum()),
            "k1_fallback": "gate=0" if k_shot == 1 else None,
        }
        return {
            "dataset": dataset,
            "k_shot": k_shot,
            "m_way": int(classes.numel()),
            "episodes": episodes,
            "class_filter": {
                "candidate_count": int(candidate_classes.numel()),
                "eligible_count": int(classes.numel()),
                "excluded": excluded,
            },
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
    parser.add_argument("--gate-bias", type=float, default=0.01)
    parser.add_argument("--gate-temperature", type=float, default=0.05)
    parser.add_argument("--output", default="ablation/results/v3-dg.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gate_temperature <= 0:
        raise ValueError("--gate-temperature must be positive")
    seed_setting(args.seed)
    evaluator = V3DGEvaluator(args.model_path, args.gpu_id, args.seed)
    results = {
        dataset: evaluator.evaluate_dataset(
            dataset,
            args.k_shot,
            args.episodes,
            args.m_way,
            args.gate_bias,
            args.gate_temperature,
        )
        for dataset in args.datasets
    }
    metrics = ("backbone", "cali_support", "support_gated")
    macro = {
        name: float(np.mean([
            result["summary"][name]["accuracy_mean"]
            for result in results.values()
        ]))
        for name in metrics
    }
    macro["fixed_cali_delta"] = macro["cali_support"] - macro["backbone"]
    macro["gated_delta"] = macro["support_gated"] - macro["backbone"]
    payload = {
        "checkpoint": args.model_path,
        "protocol": {
            "task": "federated_few_shot_domain_generalization",
            "target_parameter_updates": 0,
            "target_labels": "support-only",
            "gate_selection": "support leave-one-out",
            "gate_bias": args.gate_bias,
            "gate_temperature": args.gate_temperature,
        },
        "macro": macro,
        "datasets": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(macro, indent=2))
    for dataset, result in results.items():
        summary = result["summary"]
        print(
            f"{dataset:14s} base={summary['backbone']['accuracy_mean']:.4f} "
            f"cali={summary['cali_support']['accuracy_mean']:.4f} "
            f"gated={summary['support_gated']['accuracy_mean']:.4f} "
            f"DSG={summary['domain_specific_gain']['support_minus_reference']:+.4f} "
            f"gate={summary['gate_diagnostics']['activation_mean']:.3f}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
