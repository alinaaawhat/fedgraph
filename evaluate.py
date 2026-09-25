import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
import json
import argparse
from typing import Dict, List
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from inference.inference import PrototypeInferenceLearner
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from utils.logging import logger, timer
from utils.utils import seed_setting
import matplotlib.pyplot as plt
import seaborn as sns


class EvaluationPrototypeLearner(PrototypeInferenceLearner):
    pass

    DOWNSTREAM_DATASETS = {
        "cora": "pyg, Planetoid.Cora",
        "citeseer": "pyg, Planetoid.CiteSeer",
        "computers": "pyg, Amazon.Computers",
        "photo": "pyg, Amazon.Photo",
        "physics": "pyg, Coauthor.Physics",
        "ogbn-products": (
            "ogb.nodeproppred, PygNodePropPredDataset.ogbn-products"
        ),
    }

    def __init__(self, args):
        self._domain_embedding_cache = {}
        super().__init__(args)
        extractor = self.domain_embedder.dm_extractor
        if extractor._theta0 is None:
            extractor._theta0 = {
                key: value.detach().cpu().clone()
                for key, value in self.model_state[
                    "frozen_backbone_state_dict"
                ].items()
            }
        if self.cfg.Fingerprint.DE_type == "conv" and not hasattr(
            extractor, "_d_c_max"
        ):
            deltas = getattr(extractor, "_delta_matrices", None)
            extractor._d_c_max = (
                max(int(delta.shape[1]) for delta in deltas)
                if deltas
                else int(self.L_max)
            )

    def load_downstream_graph(self, dataset_name: str, data_path: str = None) -> Data:
        pass
        dataset_name = dataset_name.lower()
        metadata = self.cfg["_ds_meta_data"]
        if dataset_name not in metadata:
            if dataset_name not in self.DOWNSTREAM_DATASETS:
                supported = ", ".join(sorted(self.DOWNSTREAM_DATASETS))
                raise ValueError(
                    f"Unknown downstream dataset {dataset_name!r}. "
                    f"Known evaluation datasets: {supported}"
                )
            metadata[dataset_name] = self.DOWNSTREAM_DATASETS[dataset_name]

        graph = super().load_downstream_graph(dataset_name, data_path)
        if getattr(graph, "train_mask", None) is None or getattr(
            graph, "test_mask", None
        ) is None:
            self._add_stratified_masks(graph)
        return graph

    @staticmethod
    def _add_stratified_masks(graph: Data, seed: int = 42) -> None:
        pass
        labels = graph.y.view(-1).cpu()
        generator = torch.Generator().manual_seed(seed)
        masks = {
            name: torch.zeros(labels.numel(), dtype=torch.bool)
            for name in ("train_mask", "val_mask", "test_mask")
        }
        for class_id in labels[labels >= 0].unique():
            indices = (labels == class_id).nonzero(as_tuple=False).view(-1)
            indices = indices[torch.randperm(indices.numel(), generator=generator)]
            n_nodes = indices.numel()
            n_train = max(1, int(0.6 * n_nodes))
            n_val = int(0.2 * n_nodes)
            if n_nodes - n_train - n_val == 0 and n_nodes > 1:
                n_train -= 1
            masks["train_mask"][indices[:n_train]] = True
            masks["val_mask"][indices[n_train:n_train + n_val]] = True
            masks["test_mask"][indices[n_train + n_val:]] = True

        for name, mask in masks.items():
            setattr(graph, name, mask.to(graph.y.device))
        logger.info("Dataset has no official split; using a deterministic 60/20/20 split")

    def compute_domain_embedding(self, graph_data: Data) -> torch.Tensor:
        cache_key = id(graph_data)
        if cache_key in self._domain_embedding_cache:
            return self._domain_embedding_cache[cache_key]
        if self.cfg.Fingerprint.DE_type != "conv":
            embedding = super().compute_domain_embedding(graph_data)
            self._domain_embedding_cache[cache_key] = embedding
            return embedding

        extractor = self.domain_embedder.dm_extractor
        model = self.frozen_backbone
        model.load_state_dict(extractor._theta0, strict=False)
        model.to(self.device).eval()
        graph_data = graph_data.to(self.device)
        if not hasattr(graph_data, "batch") or graph_data.batch is None:
            graph_data.batch = torch.zeros(
                graph_data.num_nodes, dtype=torch.long, device=self.device
            )

        
        
        
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            node_logits, _ = model(graph_data)
            if self.cfg.Fingerprint.loss_type == "ce":
                loss = extractor.prob_loss(node_logits, graph_data.y)
            else:
                loss = extractor.prob_loss(graph_data, node_logits)
            loss.backward()

            gradient = None
            for name, parameter in model.named_parameters():
                if "weight" in name and parameter.grad is not None:
                    gradient = parameter.grad.detach().clone()
                    break
            if gradient is None:
                raise RuntimeError("Could not obtain a domain fingerprint gradient")
            if gradient.shape[0] != graph_data.x.shape[1]:
                gradient = gradient.T
            delta = -float(self.cfg.Fingerprint.probe_lr) * gradient

        target_width = int(extractor._d_c_max)
        if delta.shape[1] < target_width:
            delta = F.pad(delta, (0, target_width - delta.shape[1]))
        elif delta.shape[1] > target_width:
            delta = delta[:, :target_width]

        with torch.no_grad():
            embedding = extractor.projection(delta)
            if self.cfg.Fingerprint.l2_normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)
        embedding = embedding.detach()
        self._domain_embedding_cache[cache_key] = embedding
        return embedding


class PrototypeEvaluationAdapter:
    pass

    def __init__(self, model_path: str, device: str):
        if device.startswith("cuda"):
            gpu_id = int(device.split(":", 1)[1]) if ":" in device else 0
        else:
            raise ValueError(
                "PrototypeInferenceLearner currently requires a CUDA device"
            )
        args = SimpleNamespace(
            model_path=model_path,
            gpu_id=gpu_id,
            k_shot=5,
        )
        self.learner = EvaluationPrototypeLearner(args)
        self.graph_cache = {}

    def _graph(self, dataset_name: str, k_shot: int) -> Data:
        
        dataset_name = dataset_name.lower()
        cache_key = (dataset_name, k_shot)
        if cache_key not in self.graph_cache:
            self.learner.args.k_shot = k_shot
            self.graph_cache[cache_key] = self.learner.load_downstream_graph(
                dataset_name
            )
        return self.graph_cache[cache_key]

    @staticmethod
    def _m_way_graph(graph: Data, m_way: int | None) -> Data:
        if m_way is None:
            return graph
        classes = graph.y[graph.y >= 0].unique().sort().values[:m_way]
        node_mask = torch.isin(graph.y, classes)
        node_indices = node_mask.nonzero(as_tuple=False).view(-1)
        edge_index, _, edge_mask = subgraph(
            node_indices,
            graph.edge_index,
            relabel_nodes=True,
            num_nodes=graph.num_nodes,
            return_edge_mask=True,
        )
        result = Data(
            x=graph.x[node_indices],
            edge_index=edge_index,
            y=graph.y[node_indices],
            xe=graph.xe[edge_mask] if hasattr(graph, "xe") else None,
            batch=torch.zeros(
                node_indices.numel(),
                dtype=torch.long,
                device=graph.x.device,
            ),
        )
        for name in ("train_mask", "val_mask", "test_mask"):
            value = getattr(graph, name, None)
            if value is not None:
                setattr(result, name, value[node_indices])
        return result

    def evaluate_on_dataset(
        self,
        dataset_name: str,
        k_shot: int = 5,
        n_episodes: int = 10,
        m_way: int | None = None,
        eval_mode: str = "few-shot",
    ) -> Dict:
        if eval_mode != "few-shot":
            raise NotImplementedError(
                "PrototypeInferenceLearner has no valid zero-shot inference API"
            )
        graph = self._m_way_graph(
            self._graph(dataset_name, k_shot), m_way
        )
        result = self.learner.evaluate_multiple_runs(
            graph_data=graph,
            k_shot=k_shot,
            n_runs=n_episodes,
            distance_metric="cosine",
            use_label_prototypes=False,
        )
        if "error" in result:
            raise RuntimeError(result["error"])
        return {
            "accuracy_mean": result["mean_accuracy"],
            "accuracy_std": result["std_accuracy"],
            "details": result,
        }



class GAlignEvaluator:
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.learner = PrototypeEvaluationAdapter(model_path, device)
        self.results = {}
        
    @timer()
    def evaluate_few_shot_scaling(self, 
                                  dataset_name: str,
                                  k_shots: List[int] = [1, 3, 5, 10],
                                  n_episodes: int = 100) -> Dict:
        logger.info(f"Evaluating few-shot scaling on {dataset_name}")
        
        results = {}
        for k in k_shots:
            logger.info(f"Testing {k}-shot learning...")
            res = self.learner.evaluate_on_dataset(
                dataset_name=dataset_name,
                k_shot=k,
                n_episodes=n_episodes,
                eval_mode='few-shot'
            )
            results[f"{k}-shot"] = res
            
        return results
    
    @timer()
    def evaluate_cross_domain(self,
                            source_datasets: List[str],
                            target_datasets: List[str],
                            k_shot: int = 5,
                            n_episodes: int = 50) -> pd.DataFrame:
        logger.info("Evaluating cross-domain transfer...")
        
        results_matrix = []
        
        for target in target_datasets:
            row = {'target': target}
            
            fs_res = self.learner.evaluate_on_dataset(
                dataset_name=target,
                k_shot=k_shot,
                n_episodes=n_episodes,
                eval_mode='few-shot'
            )
            row['few_shot_acc'] = fs_res['accuracy_mean']
            row['few_shot_std'] = fs_res['accuracy_std']
            
            
            row['zero_shot_acc'] = np.nan
            row['zero_shot_std'] = np.nan
            
            row['in_pretrain'] = target in source_datasets
            
            results_matrix.append(row)
        
        return pd.DataFrame(results_matrix)
    
    @timer()
    def evaluate_m_way_scaling(self,
                              dataset_name: str,
                              m_ways: List[int] = [2, 3, 5, 7],
                              k_shot: int = 5,
                              n_episodes: int = 50) -> Dict:
        logger.info(f"Evaluating m-way scaling on {dataset_name}")
        
        results = {}
        for m in m_ways:
            logger.info(f"Testing {m}-way classification...")
            res = self.learner.evaluate_on_dataset(
                dataset_name=dataset_name,
                k_shot=k_shot,
                n_episodes=n_episodes,
                m_way=m,
                eval_mode='few-shot'
            )
            results[f"{m}-way"] = res
            
        return results
    
    def plot_results(self, save_dir: str = "evaluation_results"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if 'few_shot_scaling' in self.results:
            for dataset_name, results in self.results['few_shot_scaling'].items():
                self._plot_few_shot_scaling(
                    results, save_dir, dataset_name
                )

        if 'cross_domain' in self.results:
            self._plot_cross_domain(self.results['cross_domain'], save_dir)

        if 'm_way_scaling' in self.results:
            for dataset_name, results in self.results['m_way_scaling'].items():
                self._plot_m_way_scaling(results, save_dir, dataset_name)

    def _plot_few_shot_scaling(self, results: Dict, save_dir: str,
                               dataset_name: str):
        k_values = []
        accuracies = []
        stds = []
        
        for k_str, res in results.items():
            k = int(k_str.split('-')[0])
            k_values.append(k)
            accuracies.append(res['accuracy_mean'])
            stds.append(res['accuracy_std'])
        
        plt.figure(figsize=(8, 6))
        plt.errorbar(k_values, accuracies, yerr=stds, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Shots (k)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Few-Shot Learning Performance Scaling: {dataset_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = f"{save_dir}/few_shot_scaling_{dataset_name}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        logger.info(f"Few-shot scaling plot saved to {save_path}")
    
    def _plot_cross_domain(self, df: pd.DataFrame, save_dir: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        x = np.arange(len(df))
        width = 0.35
        
        ax1.bar(x, df['few_shot_acc'], width, yerr=df['few_shot_std'], 
                color=['green' if ip else 'blue' for ip in df['in_pretrain']],
                alpha=0.7)
        ax1.set_xlabel('Target Dataset', fontsize=12)
        ax1.set_ylabel('Few-Shot Accuracy', fontsize=12)
        ax1.set_title('Few-Shot Cross-Domain Transfer', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['target'], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        
        ax2.bar(x, df['zero_shot_acc'], width, yerr=df['zero_shot_std'],
                color=['green' if ip else 'blue' for ip in df['in_pretrain']],
                alpha=0.7)
        ax2.set_xlabel('Target Dataset', fontsize=12)
        ax2.set_ylabel('Zero-Shot Accuracy', fontsize=12)
        ax2.set_title('Zero-Shot Cross-Domain Transfer', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['target'], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='In Pretraining'),
                          Patch(facecolor='blue', alpha=0.7, label='Not in Pretraining')]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cross_domain_transfer.png", dpi=150)
        plt.close()
        
        logger.info(f"Cross-domain plot saved to {save_dir}/cross_domain_transfer.png")
    
    def _plot_m_way_scaling(self, results: Dict, save_dir: str,
                            dataset_name: str):
        m_values = []
        accuracies = []
        stds = []
        
        for m_str, res in results.items():
            m = int(m_str.split('-')[0])
            m_values.append(m)
            accuracies.append(res['accuracy_mean'])
            stds.append(res['accuracy_std'])
        
        plt.figure(figsize=(8, 6))
        plt.errorbar(m_values, accuracies, yerr=stds, marker='s', linewidth=2, markersize=8, color='red')
        plt.xlabel('Number of Classes (m-way)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'M-Way Classification Performance: {dataset_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = f"{save_dir}/m_way_scaling_{dataset_name}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        logger.info(f"M-way scaling plot saved to {save_path}")
    
    def save_results(self, save_path: str = "evaluation_results.json"):
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, pd.DataFrame):
                json_results[key] = value.to_dict('records')
            else:
                json_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")
    
    def run_comprehensive_evaluation(self,
                                    test_datasets: List[str] = None,
                                    pretrain_datasets: List[str] = None):
        logger.info("Starting comprehensive evaluation...")
        test_datasets = [
            name.lower() for name in (
                test_datasets or [
                    "cora", "citeseer", "computers", "physics", "photo"
                ]
            )
        ]
        pretrain_datasets = [
            name.lower() for name in (
                pretrain_datasets or ["pubmed", "arxiv", "wikics"]
            )
        ]

        self.results['few_shot_scaling'] = {}
        self.results['m_way_scaling'] = {}
        for dataset_name in test_datasets:
            self.results['few_shot_scaling'][dataset_name] = (
                self.evaluate_few_shot_scaling(
                    dataset_name=dataset_name,
                    k_shots=[1, 3, 5, 10],
                    n_episodes=50
                )
            )
            self.results['m_way_scaling'][dataset_name] = (
                self.evaluate_m_way_scaling(
                    dataset_name=dataset_name,
                    m_ways=[2, 3, 5],
                    k_shot=5,
                    n_episodes=30
                )
            )

        self.results['cross_domain'] = self.evaluate_cross_domain(
            source_datasets=pretrain_datasets,
            target_datasets=test_datasets,
            k_shot=5,
            n_episodes=30
        )

        self.save_results()
        self.plot_results()
        self.print_summary()
    
    def print_summary(self):
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        if 'few_shot_scaling' in self.results:
            print("\nFew-Shot Scaling Results:")
            print("-"*40)
            for dataset_name, results in self.results['few_shot_scaling'].items():
                print(f"  [{dataset_name}]")
                for k_str, res in results.items():
                    print(
                        f"    {k_str:10s}: {res['accuracy_mean']:.4f} "
                        f"± {res['accuracy_std']:.4f}"
                    )
        
        if 'cross_domain' in self.results:
            print("\nCross-Domain Transfer Results:")
            print("-"*40)
            print(self.results['cross_domain'].to_string(index=False))
        
        if 'm_way_scaling' in self.results:
            print("\nM-Way Scaling Results:")
            print("-"*40)
            for dataset_name, results in self.results['m_way_scaling'].items():
                print(f"  [{dataset_name}]")
                for m_str, res in results.items():
                    print(
                        f"    {m_str:10s}: {res['accuracy_mean']:.4f} "
                        f"± {res['accuracy_std']:.4f}"
                    )
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="G-Align Comprehensive Evaluation")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pretrained model')
    parser.add_argument('--test_datasets', nargs='+',
                       default=["cora", "citeseer", "computers", "physics",
                                "photo"],
                       help='Datasets to test on')
    parser.add_argument('--pretrain_datasets', nargs='+', 
                       default=["pubmed", "arxiv", "wikics"],
                       help='Datasets used in pretraining')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    seed_setting(args.seed)
    
    evaluator = GAlignEvaluator(args.model_path, args.device)
    
    evaluator.run_comprehensive_evaluation(
        test_datasets=args.test_datasets,
        pretrain_datasets=args.pretrain_datasets
    )


if __name__ == "__main__":
    main()