import numpy as np
import copy
import torch
import dgl
from torch.nn import KLDivLoss
from utils import set_seed
from models import InvertPPR, PropagateAPPR, MLP
from torch_geometric.data import Data
from new_utils import get_sparse_DAD_matrix
from torch_sparse.tensor import SparseTensor
from typing import Optional
from torch import Tensor
from torch_geometric.utils import subgraph
from copy import deepcopy

"""
1. Train and eval
"""


def train(model, data, feats, labels, criterion, optimizer, idx_train, lamb=1):
    """
    GNN full-batch training. Input the entire graph `g` as data.
    lamb: weight parameter lambda
    """
    model.train()

    # Compute loss and prediction
    if (
        "GCN" in model.model_name
        or "GAT" in model.model_name
        or "APPNP" in model.model_name
    ):
        _, logits = model(data, feats)
    else:
        logits = model(data, feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx_train], labels[idx_train])
    loss_val = loss.item()

    loss *= lamb
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_val


def train_sage(model, dataloader, feats, labels, criterion, optimizer, lamb=1):
    """
    Train for GraphSAGE. Process the graph in mini-batches using `dataloader` instead the entire graph `g`.
    lamb: weight parameter lambda
    """
    device = feats.device
    model.train()
    total_loss = 0
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_labels = labels[output_nodes]

        # Compute loss and prediction
        logits = model(blocks, batch_feats)
        out = logits.log_softmax(dim=1)
        loss = criterion(out, batch_labels)
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def train_mini_batch(model, feats, labels, batch_size, criterion, optimizer, lamb=1):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    model.train()
    num_batches = max(1, feats.shape[0] // batch_size)
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        _, logits = model(None, feats[idx_batch[i]])
        out = logits.log_softmax(dim=1)

        loss = criterion(out, labels[idx_batch[i]])
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches


def get_PGD_inputs(model, feats, labels, criterion, args):
    iters = 5
    eps = args.adv_eps
    alpha = eps / 4

    # init
    delta = torch.rand(feats.shape) * eps * 2 - eps
    delta = delta.to(feats.device)
    delta = torch.nn.Parameter(delta)

    for _ in range(iters):
        p_feats = feats + delta

        _, logits = model(None, p_feats)
        out = logits.log_softmax(dim=1)

        loss = criterion(out, labels)
        loss.backward()

        # delta update
        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)

    return delta.detach()


def train_distillation_batch(
    model, feats, labels, batch_size, criterion, optimizer, lamb=1
):
    model.train()
    num_batches = max(1, feats.shape[0] // batch_size)
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        _, logits = model(None, feats[idx_batch[i]])
        out = logits.log_softmax(dim=1)
        loss_label = criterion(out, labels[idx_batch[i]])
        loss = loss_label
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches


def train_student_InD_tran(
    mlp: MLP,
    feats: torch.Tensor,
    mlp_optimizer: torch,
    norm_adj: torch.Tensor,
    prop: InvertPPR,
    data: Data,
    out_t_all: torch.Tensor,
    batch_size: int,
    criterion: KLDivLoss,
    args,
    lamb: float = 0.0,
    device: torch.device = "cuda",
):
    mlp.train()
    mlp_optimizer.zero_grad()

    num_nodes = data.x.shape[0]
    num_batches = max(1, num_nodes // batch_size)
    idx_batch = torch.randperm(num_nodes)[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        _, out = mlp(None, feats.to(device))
        # Invert
        out_l = (1 / (1 - args.gamma)) * prop(
            logits=out,
            edge_index=norm_adj,
            self_coeff=args.self_coeff,
            gamma=args.gamma,
        )[idx_batch[i]]

        loss = criterion(out_l.log_softmax(dim=-1), out_t_all.to(device)[idx_batch[i]])

        total_loss += loss.item()
        loss *= lamb
        mlp_optimizer.zero_grad()
        loss.backward()
        mlp_optimizer.step()
    return total_loss / num_batches
    # total_loss = 0
    # for i in range(num_batches):
    #     _, out = mlp(None, data.x.to(device))
    #     # Invert
    #     if args.is_InD:
    #         out_l = (1 / (1 - args.gamma)) * prop(
    #             logits=out,
    #             edge_index=norm_adj,
    #             self_coeff=args.self_coeff,
    #             gamma=args.gamma,
    #         )[idx_batch[i]]
    #     if args.is_PnD:
    #         training_idx = idx_train if args.fix_train else None
    #         if args.softmax_train:
    #             out_l = prop(
    #                 logits=out.softmax(dim=1),
    #                 edge_index=norm_adj,
    #                 iteration_num=args.prop_iteration,
    #                 training_idx=training_idx,
    #             )[idx_batch[i]].log()
    #         else:
    #             out_l = prop(
    #                 logits=out,
    #                 edge_index=norm_adj,
    #                 iteration_num=args.prop_iteration,
    #                 training_idx=training_idx,
    #             )[idx_batch[i]]

    #     loss = criterion(out_l.log_softmax(dim=-1), out_t_all.to(device)[idx_batch[i]])
    #     total_loss += loss.item()
    #     loss *= lamb
    #     mlp_optimizer.zero_grad()
    #     loss.backward()
    #     mlp_optimizer.step()
    # return total_loss / num_batches


def train_student_PnD_tran(
    mlp: MLP,
    feats: torch.Tensor,
    mlp_optimizer: torch,
    out_t_all_prop: torch.Tensor,
    data: Data,
    batch_size: int,
    criterion: KLDivLoss,
    lamb: float = 0.0,
    device: torch.device = "cuda",
):
    mlp.train()
    mlp_optimizer.zero_grad()

    num_nodes = data.x.shape[0]
    num_batches = max(1, num_nodes // batch_size)
    idx_batch = torch.randperm(num_nodes)[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        _, out = mlp(None, feats.to(device))
        loss = criterion(
            out[idx_batch[i], :].log_softmax(dim=-1),
            out_t_all_prop[idx_batch[i]],
        )

        total_loss += loss.item()
        loss *= lamb
        mlp_optimizer.zero_grad()
        loss.backward()
        mlp_optimizer.step()
    return total_loss / num_batches


def train_student_InD_ind(
    mlp: MLP,
    feats: torch.Tensor,
    mlp_optimizer: torch,
    norm_adj: torch.Tensor,
    prop: InvertPPR,
    data: Data,
    out_t_all: torch.Tensor,
    batch_size: int,
    criterion: KLDivLoss,
    args,
    lamb: float = 0.0,
    device: torch.device = "cuda",
):
    mlp.train()
    mlp_optimizer.zero_grad()

    num_nodes = data.idx_obs.nelement()
    idx_obs = data.idx_obs.to(device)
    idx = torch.randperm(num_nodes)
    idx_obs = idx_obs.view(-1)[idx].view(idx_obs.size())
    num_batches = max(1, num_nodes // batch_size)
    idx_batch = idx_obs[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        _, out = mlp(None, feats.to(device))
        # Invert
        out_l = (1 / (1 - args.gamma)) * prop(
            logits=out,
            edge_index=norm_adj,
            self_coeff=args.self_coeff,
            gamma=args.gamma,
        )[idx_batch[i]]

        loss = criterion(out_l.log_softmax(dim=-1), out_t_all.to(device)[idx_batch[i]])

        total_loss += loss.item()
        loss *= lamb
        mlp_optimizer.zero_grad()
        loss.backward()
        mlp_optimizer.step()
    return total_loss / num_batches


def train_student_PnD_ind(
    mlp: MLP,
    feats: torch.Tensor,
    mlp_optimizer: torch,
    out_t_all_prop: torch.Tensor,
    data: Data,
    batch_size: int,
    criterion: KLDivLoss,
    lamb: float = 0.0,
    device: torch.device = "cuda",
):
    mlp.train()
    mlp_optimizer.zero_grad()

    num_nodes = data.idx_obs.nelement()
    idx_obs = data.idx_obs.to(device)
    idx = torch.randperm(num_nodes)
    idx_obs = idx_obs.view(-1)[idx].view(idx_obs.size())
    num_batches = max(1, num_nodes // batch_size)
    idx_batch = idx_obs[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        _, out = mlp(None, feats[idx_batch[i], :].to(device))
        # Invert
        loss = criterion(
            out.log_softmax(dim=-1),
            out_t_all_prop[idx_batch[i], :],
        )

        total_loss += loss.item()
        loss *= lamb
        mlp_optimizer.zero_grad()
        loss.backward()
        mlp_optimizer.step()
    return total_loss / num_batches


def evaluate(model, data, feats, labels, criterion, evaluator, idx_eval=None):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    model.eval()
    with torch.no_grad():
        if (
            "GCN" in model.model_name
            or "GAT" in model.model_name
            or "APPNP" in model.model_name
        ):
            emb_list, logits = model.inference(data, feats)
        elif "LAGE" in model.model_name or "CAGE" in model.model_name:
            logits = model.inference(data, feats)
            emb_list = None
        else:
            logits, emb_list = model.inference(data, feats)
        out = logits.log_softmax(dim=1)
        if idx_eval is None:
            loss = criterion(out, labels)
            score = evaluator(out, labels)
        else:
            loss = criterion(out[idx_eval], labels[idx_eval])
            score = evaluator(out[idx_eval], labels[idx_eval])
    return out, loss.item(), score, emb_list


def evaluate_mini_batch(
    model, feats, labels, criterion, batch_size, evaluator, idx_eval=None
):
    """
    Evaluate MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    Return:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """

    model.eval()
    with torch.no_grad():
        num_batches = int(np.ceil(len(feats) / batch_size))
        out_list = []
        for i in range(num_batches):
            _, logits = model.inference(
                None, feats[batch_size * i : batch_size * (i + 1)]
            )
            out = logits.log_softmax(dim=1)
            out_list += [out.detach()]

        out_all = torch.cat(out_list)

        if idx_eval is None:
            loss = criterion(out_all, labels)
            score = evaluator(out_all, labels)
        else:
            loss = criterion(out_all[idx_eval], labels[idx_eval])
            score = evaluator(out_all[idx_eval], labels[idx_eval])

    return out_all, loss.item(), score


"""
2. Run teacher
"""


def run_transductive(
    conf,
    model,
    g,
    feats,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    """
    Train and eval under the transductive setting.
    The train/valid/test split is specified by `indices`.
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]

    idx_train, idx_val, idx_test = indices

    feats = feats.to(device)
    labels = labels.to(device)

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves memory and CPU.
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        # SAGE inference is implemented as layer by layer, so the full-neighbor sampler only collects one-hop neighors
        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader_eval = dgl.dataloading.NodeDataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        data = dataloader
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_train, labels_train = feats[idx_train], labels[idx_train]
        feats_val, labels_val = feats[idx_val], labels[idx_val]
        feats_test, labels_test = feats[idx_test], labels[idx_test]
    else:
        g = g.to(device)
        data = g
        data_eval = g

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            # with data.enable_cpu_affinity():
            loss = train_sage(model, data, feats, labels, criterion, optimizer)
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer
            )
        else:
            loss = train(model, data, feats, labels, criterion, optimizer, idx_train)

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, labels_train, criterion, batch_size, evaluator
                )
                _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, criterion, batch_size, evaluator
                )
                _, loss_test, score_test = evaluate_mini_batch(
                    model, feats_test, labels_test, criterion, batch_size, evaluator
                )
            else:
                # with dataloader_eval.enable_cpu_affinity():
                out, loss_train, score_train, emb_list = evaluate(
                    model, data_eval, feats, labels, criterion, evaluator, idx_train
                )
                # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
                loss_val = criterion(out[idx_val], labels[idx_val]).item()
                score_val = evaluator(out[idx_val], labels[idx_val])
                loss_test = criterion(out[idx_test], labels[idx_test]).item()
                score_test = evaluator(out[idx_test], labels[idx_test])

            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test,
                    score_train,
                    score_val,
                    score_test,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        out, _, score_val = evaluate_mini_batch(
            model, feats, labels, criterion, batch_size, evaluator, idx_val
        )
        emb_list = None
    else:
        # with dataloader_eval.enable_cpu_affinity():
        out, _, score_val, emb_list = evaluate(
            model, data_eval, feats, labels, criterion, evaluator, idx_val
        )

    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test, emb_list


def run_inductive(
    conf,
    model,
    g,
    feats,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    """
    Train and eval under the inductive setting.
    The train/valid/test split is specified by `indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    loss_and_score: Stores losses and scores.
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices

    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_g = g.subgraph(idx_obs)

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        obs_g.create_formats_()
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        obs_dataloader = dgl.dataloading.NodeDataLoader(
            obs_g,
            obs_idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        obs_dataloader_eval = dgl.dataloading.NodeDataLoader(
            obs_g,
            torch.arange(obs_g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )
        dataloader_eval = dgl.dataloading.NodeDataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        obs_data = obs_dataloader
        obs_data_eval = obs_dataloader_eval
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_train, labels_train = obs_feats[obs_idx_train], obs_labels[obs_idx_train]
        feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
        feats_test_tran, labels_test_tran = (
            obs_feats[obs_idx_test],
            obs_labels[obs_idx_test],
        )
        feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    else:
        obs_g = obs_g.to(device)
        g = g.to(device)

        obs_data = obs_g
        obs_data_eval = obs_g
        data_eval = g

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            # with obs_dataloader.enable_cpu_affinity():
            loss = train_sage(
                model, obs_data, obs_feats, obs_labels, criterion, optimizer
            )
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer
            )
        else:
            loss = train(
                model,
                obs_data,
                obs_feats,
                obs_labels,
                criterion,
                optimizer,
                obs_idx_train,
            )

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, labels_train, criterion, batch_size, evaluator
                )
                _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, criterion, batch_size, evaluator
                )
                _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                    model,
                    feats_test_tran,
                    labels_test_tran,
                    criterion,
                    batch_size,
                    evaluator,
                )
                _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                    model,
                    feats_test_ind,
                    labels_test_ind,
                    criterion,
                    batch_size,
                    evaluator,
                )
            else:
                # with obs_dataloader_eval.enable_cpu_affinity():
                obs_out, loss_train, score_train, emb_list = evaluate(
                    model,
                    obs_data_eval,
                    obs_feats,
                    obs_labels,
                    criterion,
                    evaluator,
                    obs_idx_train,
                )
                # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
                loss_val = criterion(
                    obs_out[obs_idx_val], obs_labels[obs_idx_val]
                ).item()
                score_val = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
                loss_test_tran = criterion(
                    obs_out[obs_idx_test], obs_labels[obs_idx_test]
                ).item()
                score_test_tran = evaluator(
                    obs_out[obs_idx_test], obs_labels[obs_idx_test]
                )

                # Evaluate the inductive part with the full graph
                # with dataloader_eval.enable_cpu_affinity():
                out, loss_test_ind, score_test_ind, emb_list = evaluate(
                    model,
                    data_eval,
                    feats,
                    labels,
                    criterion,
                    evaluator,
                    idx_test_ind,
                )
            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_train,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]
            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        obs_out, _, score_val = evaluate_mini_batch(
            model, obs_feats, obs_labels, criterion, batch_size, evaluator, obs_idx_val
        )
        out, _, score_test_ind = evaluate_mini_batch(
            model, feats, labels, criterion, batch_size, evaluator, idx_test_ind
        )

    else:
        # with obs_dataloader_eval.enable_cpu_affinity():
        obs_out, _, score_val, emb_list = evaluate(
            model,
            obs_data_eval,
            obs_feats,
            obs_labels,
            criterion,
            evaluator,
            obs_idx_val,
        )
        # with dataloader_eval.enable_cpu_affinity():
        out, _, score_test_ind, emb_list = evaluate(
            model, data_eval, feats, labels, criterion, evaluator, idx_test_ind
        )

    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out[idx_obs] = obs_out
    logger.info(
        f"Best valid model at epoch: {best_epoch :3d}, score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    if "MLP" in model.model_name:  # used in train_teacher with MLP as teacher model
        return out, score_val, score_test_tran, score_test_ind, None
    return out, score_val, score_test_tran, score_test_ind, emb_list


"""
3. Distill
"""


def distill_run_transductive(
    conf,
    model,
    feats,
    labels,
    out_t_all,
    distill_indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
    g,
    args,
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    idx_l, idx_t, idx_val, idx_test = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)

    feats_l, labels_l = feats[idx_l], labels[idx_l]
    # feats_t, out_t = feats[idx_t], out_t_all[idx_t]
    feats_val, labels_val = feats[idx_val], labels[idx_val]
    feats_test, labels_test = feats[idx_test], labels[idx_test]

    best_epoch, best_score_val, count = 0, 0, 0

    row, col = g.edges()
    edge_index = torch.vstack([row, col])

    data = Data(edge_index=edge_index)
    data.x = feats
    data = data.to(device)

    norm_adj: SparseTensor = get_sparse_DAD_matrix(data=data, device=device)
    if args.is_InD:
        prop = InvertPPR()
    if args.is_PnD:
        prop = PropagateAPPR(num_iteration=args.prop_iteration, gamma=args.gamma)
        training_idx = idx_l.to(device) if args.fix_train else None
        out_t_all_prop = prop(
            logits=out_t_all.softmax(dim=1),
            edge_index=norm_adj,
            training_idx=training_idx,
        )

    # g.create_formats_()
    # The full-neighbor one-hop neighors is enough
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    # dataloader = dgl.dataloading.NodeDataLoader(
    #     g,
    #     torch.arange(g.num_nodes(), device=g.device),
    #     sampler,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=conf["num_workers"],
    # )
    # data = g

    for epoch in range(1, conf["max_epoch"] + 1):
        # Label train
        loss_l = train_distillation_batch(
            model=model,
            feats=feats_l,
            labels=labels_l,
            batch_size=batch_size,
            criterion=criterion_l,
            optimizer=optimizer,
            lamb=1 - lamb,
        )
        # Distill train
        if args.is_InD:
            loss_t = train_student_InD_tran(
                mlp=model,
                feats=feats,
                mlp_optimizer=optimizer,
                norm_adj=norm_adj,
                prop=prop,
                data=data,
                out_t_all=out_t_all,
                criterion=criterion_t,
                batch_size=args.batch_size,
                args=args,
                lamb=lamb,
            )
        if args.is_PnD:
            loss_t = train_student_PnD_tran(
                mlp=model,
                feats=feats,
                mlp_optimizer=optimizer,
                out_t_all_prop=out_t_all_prop,
                data=data,
                criterion=criterion_t,
                batch_size=args.batch_size,
                lamb=lamb,
            )

        loss = loss_l + loss_t
        # loss = loss_t

        if epoch % conf["eval_interval"] == 0:
            _, loss_l, score_l = evaluate_mini_batch(
                model=model,
                feats=feats_l,
                labels=labels_l,
                criterion=criterion_l,
                batch_size=batch_size,
                evaluator=evaluator,
            )
            _, loss_val, score_val = evaluate_mini_batch(
                model=model,
                feats=feats_val,
                labels=labels_val,
                criterion=criterion_l,
                batch_size=batch_size,
                evaluator=evaluator,
            )
            _, loss_test, score_test = evaluate_mini_batch(
                model=model,
                feats=feats_test,
                labels=labels_test,
                criterion=criterion_l,
                batch_size=batch_size,
                evaluator=evaluator,
            )

            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
            loss_and_score += [
                [epoch, loss_l, loss_val, loss_test, score_l, score_val, score_test]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    out, _, score_val = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_val
    )
    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test = evaluator(out[idx_test], labels_test)

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test


def distill_run_inductive(
    conf,
    model,
    feats,
    labels,
    out_t_all,
    # out_emb_t_all,
    distill_indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
    g,
    args,
):
    """
    Distill training and eval under the inductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively.
    loss_and_score: Stores losses and scores.
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    (
        obs_idx_l,
        obs_idx_t,
        obs_idx_val,
        obs_idx_test,
        idx_obs,
        idx_test_ind,
    ) = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_out_t = out_t_all[idx_obs]

    feats_l, labels_l = obs_feats[obs_idx_l], obs_labels[obs_idx_l]
    feats_t, out_t = obs_feats[obs_idx_t], obs_out_t[obs_idx_t]
    # out_emb_t = out_emb_t_all[obs_idx_t]
    # out_emb_l = out_emb_t_all[obs_idx_l]
    feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, labels_test_tran = (
        obs_feats[obs_idx_test],
        obs_labels[obs_idx_test],
    )
    feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    best_epoch, best_score_val, count = 0, 0, 0

    row, col = g.edges()
    edge_index = torch.vstack([row, col])

    obs_data = Data(edge_index=edge_index)
    obs_data.idx_obs = idx_obs
    # obs_data.obs_idx_l = obs_idx_l
    # obs_data.obs_idx_t = obs_idx_t
    # obs_data.obs_idx_val = obs_idx_val
    # obs_data.obs_idx_test = obs_idx_test
    # obs_data.idx_test_ind = idx_test_ind
    obs_edge_index, _ = subgraph(
        subset=idx_obs,
        edge_index=edge_index,
        relabel_nodes=False,
        num_nodes=feats.shape[0],
    )
    obs_data.x = feats
    obs_data.edge_index = obs_edge_index
    obs_data = obs_data.to(device)

    norm_adj: SparseTensor = get_sparse_DAD_matrix(data=obs_data, device=device)
    if args.is_InD:
        prop = InvertPPR()
    if args.is_PnD:
        prop = PropagateAPPR(num_iteration=args.prop_iteration, gamma=args.gamma)
        training_idx = obs_idx_l.to(device) if args.fix_train else None
        out_t_all_prop = prop(
            logits=out_t_all.softmax(dim=-1),
            edge_index=norm_adj,
            training_idx=training_idx,
        )

    for epoch in range(1, conf["max_epoch"] + 1):
        loss_l = train_distillation_batch(
            model=model,
            feats=feats_l,
            labels=labels_l,
            batch_size=batch_size,
            criterion=criterion_l,
            optimizer=optimizer,
            lamb=1 - lamb,
        )
        # Distill train
        if args.is_PnD:
            loss_t = train_student_PnD_ind(
                mlp=model,
                feats=feats,
                mlp_optimizer=optimizer,
                data=obs_data,
                out_t_all_prop=out_t_all_prop,
                criterion=criterion_t,
                batch_size=args.batch_size,
                lamb=lamb,
            )
        elif args.is_InD:
            loss_t = train_student_InD_ind(
                mlp=model,
                feats=feats,
                mlp_optimizer=optimizer,
                norm_adj=norm_adj,
                prop=prop,
                data=obs_data,
                out_t_all=out_t_all,
                criterion=criterion_t,
                batch_size=args.batch_size,
                args=args,
                lamb=lamb,
            )
        else:
            loss_t = train_distillation_batch(
                model=model,
                feats=feats_t,
                labels=obs_out_t,
                batch_size=batch_size,
                criterion=criterion_t,
                optimizer=optimizer,
                lamb=lamb,
            )
        loss = loss_l + loss_t

        if epoch % conf["eval_interval"] == 0:
            _, loss_l, score_l = evaluate_mini_batch(
                model=model,
                feats=feats_l,
                labels=labels_l,
                criterion=criterion_l,
                batch_size=batch_size,
                evaluator=evaluator,
            )
            _, loss_val, score_val = evaluate_mini_batch(
                model=model,
                feats=feats_val,
                labels=labels_val,
                criterion=criterion_l,
                batch_size=batch_size,
                evaluator=evaluator,
            )
            _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                model=model,
                feats=feats_test_tran,
                labels=labels_test_tran,
                criterion=criterion_l,
                batch_size=batch_size,
                evaluator=evaluator,
            )
            _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                model=model,
                feats=feats_test_ind,
                labels=labels_test_ind,
                criterion=criterion_l,
                batch_size=batch_size,
                evaluator=evaluator,
            )

            logger.debug(
                f"Ep {epoch:3d} | l: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_l,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_l,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    obs_out, _, score_val = evaluate_mini_batch(
        model, obs_feats, obs_labels, criterion_l, batch_size, evaluator, obs_idx_val
    )
    out, _, score_test_ind = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_test_ind
    )

    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test_tran = evaluator(obs_out[obs_idx_test], labels_test_tran)
    out[idx_obs] = obs_out

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d} score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind
