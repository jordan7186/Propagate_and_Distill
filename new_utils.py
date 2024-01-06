from typing import Optional, Tuple
from pathlib import Path
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
import numpy as np
import dataloader
from torch_geometric.utils import subgraph
from torch_geometric.nn import MLP, GCN
from copy import deepcopy
from torch_geometric.utils import scatter
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.transforms import ToSparseTensor
import yaml

"""
Dataset loading and processing etc.
"""


def load_out_t(path):
    return torch.from_numpy(np.load(path.joinpath("out.npz"))["arr_0"])


def load_idx_and_outputs(
    dataset_name: str,
    mode: int,
    seed: int,
    teacher_model: str = "SAGE",
    split_rate: Optional[float] = None,
) -> Tuple:
    assert mode in {"transductive", "inductive"}

    base_path = Path.cwd()
    if mode == "inductive":
        assert split_rate is not None, "split_rate not defined"
        deep_t_path = base_path.joinpath(
            "outputs",
            mode,
            f"split_rate_{split_rate}",
            dataset_name,
            f"{teacher_model}_MLP",
            f"seed_{seed}",
        )

    elif mode == "transductive":
        deep_t_path = base_path.joinpath(
            "outputs", mode, dataset_name, f"{teacher_model}_MLP", f"seed_{seed}"
        )
    return base_path, deep_t_path


def return_dataset_from_DGL(
    dataset_name: str,
    seed: int,
    labelrate_train: int = 20,
    labelrate_val: int = 30,
    normalize: bool = True,
) -> Data:
    base_path = Path.cwd()
    graph, labels, _, _, _ = dataloader.load_data(
        dataset=dataset_name,
        dataset_path=base_path.joinpath("data"),
        seed=seed,
        labelrate_train=labelrate_train,
        labelrate_val=labelrate_val,
    )
    data_x = graph.ndata["feat"]
    if normalize:
        data_x /= data_x.sum(dim=1).view(-1, 1)
    src, dst = graph.edges()

    return Data(edge_index=torch.vstack([src, dst]), x=data_x, y=labels)


def load_legacy_data(
    dataset_name: str,
    mode: int,
    teacher_model: str,
    seed: int,
    split_rate: Optional[float] = None,
) -> Tuple:
    _, deep_t_path = load_idx_and_outputs(
        dataset_name=dataset_name,
        mode=mode,
        teacher_model=teacher_model,
        seed=seed,
        split_rate=split_rate,
    )
    """
    Transductive: distill_indices = (idx_l, idx_t, idx_val, idx_test)
    Inductive: distill_indices = (obs_idx_l, obs_idx_t, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind)
    """
    distill_indices = torch.load(deep_t_path.joinpath("idx_splits.pt"))
    out_t_all = load_out_t(path=deep_t_path)
    graph = return_dataset_from_DGL(dataset_name=dataset_name, seed=seed)

    return distill_indices, out_t_all, graph


def prep_for_transductive_data(distill_indices: Tuple, graph: Data) -> Data:
    idx_l, idx_t, idx_val, idx_test = distill_indices
    graph.idx_l = idx_l
    graph.idx_t = idx_t
    graph.idx_val = idx_val
    graph.idx_test = idx_test

    return graph


import torch_geometric as pyg
import dgl


# def prep_for_inductive_data(distill_indices: Tuple, graph: Data) -> Data:
#     (
#         obs_idx_l,
#         obs_idx_t,
#         obs_idx_val,
#         obs_idx_test,
#         idx_obs,
#         idx_test_ind,
#     ) = distill_indices

#     g_nx = dgl.to_networkx(graph.cpu())
#     data = pyg.utils.from_networkx(g_nx)
#     data.x = graph.ndata["feat"]

#     data.obs_idx_l = obs_idx_l
#     data.obs_idx_t = obs_idx_t
#     data.obs_idx_val = obs_idx_val
#     data.obs_idx_test = obs_idx_test
#     data.idx_obs = idx_obs
#     data.idx_test_ind = idx_test_ind

#     data.obs_edge_index = subgraph(
#         subset=idx_obs, edge_index=data.edge_index, relabel_nodes=False
#     )

#     return data


"""
Model initialization etc.
"""


def get_training_config(config_path, model_name, dataset):
    with open(config_path, "r") as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    dataset_specific_config = full_config["global"]
    model_specific_config = full_config[dataset][model_name]

    if model_specific_config is not None:
        specific_config = dict(dataset_specific_config, **model_specific_config)
    else:
        specific_config = dataset_specific_config

    specific_config["model_name"] = model_name
    return specific_config


def grab_new_MLP_with_opt(dataset, input_dim, output_dim, mode, device: str = "cuda"):
    assert mode in {
        "transductive",
        "inductive",
    }, "mode is either transductive or inductive"
    if mode == "transductive":
        config_path = Path.cwd().joinpath("tran.conf.yaml")
    else:
        config_path = Path.cwd().joinpath("ind.conf.yaml")
    conf = get_training_config(
        config_path=config_path, model_name="MLP", dataset=dataset
    )
    mlp = MLP(
        [input_dim, int(conf["hidden_dim"]), output_dim],
        dropout=conf["dropout_ratio"],
        norm=None,
    ).to(device)
    mlp_optimizer = torch.optim.Adam(
        mlp.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"]
    )
    return mlp, mlp_optimizer, conf


class EarlyStopper:
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_val: float = 0
        self.test_acc: float = 0
        self.count: int = 0
        self.model = None

    def is_stopping(
        self, curr_val: float, curr_test: float, epoch: int, model: Optional[GCN] = None
    ):
        if curr_val >= self.best_val:
            self.best_val = curr_val
            self.test_acc = curr_test
            self.count = 0
            self.epoch = epoch
            self.model = deepcopy(model)
        else:
            self.count += 1
            # If triggers early stopping
            if self.count > self.patience:
                return True


def get_teacher_performance(
    out_t_all: torch.Tensor, test_ind: torch.Tensor, label: torch.Tensor
) -> None:
    pred = out_t_all[test_ind].argmax(dim=1)
    return 100 * torch.sum(pred == label[test_ind]) / len(test_ind)


# def get_norm_adj(
#     data: Data, self_loops: bool = True, num_nodes: Optional[int] = None
# ) -> torch.Tensor:
#     # Just a symmetrically normalized ADJACENCY matrix
#     edge_index_ = data.edge_index
#     if self_loops:
#         edge_index_ = add_self_loops(remove_self_loops(edge_index_)[0])[0]
#     row, col = data.edge_index
#     if num_nodes is None:
#         num_nodes = data.x.shape[0]
#     edge_weight = torch.ones(data.edge_index.size(1), device=data.edge_index.device)
#     deg = scatter(
#         edge_weight,
#         row,
#         0,
#         dim_size=num_nodes,
#         reduce="sum",
#     )
#     deg_inv_sqrt = deg.pow_(-0.5)
#     deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
#     edge_weight_norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
#     return torch.sparse_coo_tensor(
#         data.edge_index, edge_weight_norm, (num_nodes, num_nodes)
#     )


sparsifier = ToSparseTensor()


def get_sparse_DAD_matrix(data: Data, device: torch.device = "cuda"):
    data = sparsifier(data)
    adj_t = data.adj_t.to(device)
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    return deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
