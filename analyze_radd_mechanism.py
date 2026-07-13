"""
Mechanism diagnostics for RADD.

This script does not train models. It loads existing pruned models and produces
CSV summaries for explanatory figures:
  1) exposure counts by memorization-risk bin;
  2) member/non-member confidence, entropy, margin, and loss gaps by risk bin;
  3) teacher-student KL by risk bin.
"""
import argparse
import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from base_model import BaseModel
from data_process import load_mem_scores, risk_budget_window, slide_window
from datasets import get_dataset


parser = argparse.ArgumentParser(description="RADD mechanism diagnostics")
parser.add_argument("device", default=0, type=int, help="GPU id to use")
parser.add_argument("config_path", type=str, help="config file path")
parser.add_argument("--dataset_name", default="cifar100", type=str)
parser.add_argument("--model_name", default="resnet18", type=str)
parser.add_argument("--num_cls", default=100, type=int)
parser.add_argument("--input_dim", default=3, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--pruner_name", default="iter_pruning", type=str)
parser.add_argument("--prune_sparsity", default=0.6, type=float)
parser.add_argument("--prune_iter", default=5, type=int)
parser.add_argument("--prune_epochs", default=21, type=int)
parser.add_argument("--width", default=100, type=int)
parser.add_argument("--stride", default=5, type=int)
parser.add_argument("--mem_thre", default=0.6, type=float)
parser.add_argument("--high_risk_cap", default=1, type=int)
parser.add_argument("--high_risk_ratio", default=0.1, type=float)
parser.add_argument("--distill_temp", default=3.0, type=float)
parser.add_argument("--methods", default="base,swmr,radd", type=str)
parser.add_argument("--num_bins", default=5, type=int)
parser.add_argument("--output_dir", default="", type=str)


class IndexedDataset(Dataset):
    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        self.indices = list(indices) if indices is not None else None

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.dataset)

    def __getitem__(self, item):
        if self.indices is None:
            x, y = self.dataset[item]
            return x, y, -1
        data_index = self.indices[item]
        x, y = self.dataset[data_index]
        return x, y, int(data_index)


def model_prefix(method, pruner_name, sparsity):
    if method == "base":
        return "{}_{}".format(pruner_name, sparsity)
    if method == "rsw":
        return "{}_{}_slide_re".format(pruner_name, sparsity)
    if method == "rmr":
        return "{}_{}_ml2".format(pruner_name, sparsity)
    if method == "swmr":
        return "{}_{}_slide_ml2".format(pruner_name, sparsity)
    if method == "radd":
        return "{}_{}_risk_distill_swmr".format(pruner_name, sparsity)
    raise ValueError("Unsupported method: {}".format(method))


def load_pruned_model(args, device, save_folder, method):
    prefix = model_prefix(method, args.pruner_name, args.prune_sparsity)
    model_dir = os.path.join(save_folder, prefix + "_model")
    model_path = os.path.join(model_dir, "best.pth")
    if not os.path.isfile(model_path):
        print("Skip method {}: missing {}".format(method, model_path))
        return None, model_dir
    bm = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    bm.model.load_state_dict(torch.load(model_path, map_location=device))
    bm.model.eval()
    return bm.model, model_dir


def evaluate_dataset(model, loader, device, num_cls, teacher_model=None, temp=3.0):
    rows = []
    model.eval()
    if teacher_model is not None:
        teacher_model.eval()
    with torch.no_grad():
        for inputs, targets, data_indices in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            log_probs = F.log_softmax(outputs, dim=1)
            conf, pred = probs.max(dim=1)
            sorted_probs, _ = probs.sort(dim=1, descending=True)
            if num_cls > 1:
                margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            else:
                margin = torch.zeros_like(conf)
            entropy = -(probs * log_probs).sum(dim=1)
            ce_loss = F.cross_entropy(outputs, targets, reduction="none")
            correct = pred.eq(targets).float()

            if teacher_model is None:
                teacher_kl = torch.full_like(conf, np.nan)
            else:
                teacher_outputs = teacher_model(inputs)
                t = max(float(temp), 1e-6)
                teacher_probs = F.softmax(teacher_outputs / t, dim=1)
                student_log_probs = F.log_softmax(outputs / t, dim=1)
                teacher_kl = F.kl_div(
                    student_log_probs, teacher_probs, reduction="none"
                ).sum(dim=1) * (t * t)

            batch = torch.stack([
                conf.detach().cpu(),
                entropy.detach().cpu(),
                margin.detach().cpu(),
                ce_loss.detach().cpu(),
                correct.detach().cpu(),
                teacher_kl.detach().cpu(),
            ], dim=1).numpy()
            for idx, values in zip(data_indices.numpy(), batch):
                rows.append({
                    "data_index": int(idx),
                    "confidence": float(values[0]),
                    "entropy": float(values[1]),
                    "margin": float(values[2]),
                    "loss": float(values[3]),
                    "correct": float(values[4]),
                    "teacher_kl": float(values[5]),
                })
    return pd.DataFrame(rows)


def assign_risk_bins(member_indices, mem_scores, num_bins):
    member_scores = np.asarray([mem_scores[int(idx)] for idx in member_indices], dtype=np.float32)
    ranks = pd.Series(member_scores).rank(method="first").values
    bins = np.ceil(ranks / len(member_scores) * num_bins).astype(int)
    bins = np.clip(bins, 1, num_bins)
    return dict((int(idx), int(bin_id)) for idx, bin_id in zip(member_indices, bins))


def aggregate_metric_gaps(member_df, nonmember_df, bin_map, mem_scores, method, args):
    non = nonmember_df.mean(numeric_only=True)
    rows = []
    member_df = member_df.copy()
    member_df["risk_bin"] = member_df["data_index"].map(bin_map)
    member_df["memscore"] = member_df["data_index"].map(lambda x: float(mem_scores[int(x)]))
    for bin_id, group in member_df.groupby("risk_bin"):
        row = {
            "dataset": args.dataset_name,
            "architecture": args.model_name,
            "method": method,
            "risk_bin": int(bin_id),
            "n_member": int(len(group)),
            "memscore_mean": float(group["memscore"].mean()),
            "memscore_min": float(group["memscore"].min()),
            "memscore_max": float(group["memscore"].max()),
        }
        for metric in ["confidence", "entropy", "margin", "loss", "correct", "teacher_kl"]:
            member_val = float(group[metric].mean())
            non_val = float(non[metric]) if metric in non.index else np.nan
            row["member_" + metric] = member_val
            row["nonmember_" + metric] = non_val
            row[metric + "_gap"] = member_val - non_val
        rows.append(row)
    return rows


def exposure_rows(args, victim_train_list, mem_scores, bin_map):
    total_finetune_epochs = args.prune_epochs * args.prune_iter
    rows = []

    method_counts = {}
    method_counts["base"] = Counter(dict((int(idx), total_finetune_epochs) for idx in victim_train_list))

    swmr_windows = slide_window(
        args.dataset_name, args.model_name, args.prune_epochs,
        victim_train_list, args.width, args.stride
    )
    swmr_counter = Counter()
    for epoch_indices in swmr_windows:
        for idx in epoch_indices:
            swmr_counter[int(idx)] += args.prune_iter
    method_counts["swmr"] = swmr_counter

    radd_windows = risk_budget_window(
        args.dataset_name, args.model_name, total_finetune_epochs,
        victim_train_list, args.width, args.stride, args.mem_thre,
        args.high_risk_cap, args.high_risk_ratio
    )
    radd_counter = Counter()
    for epoch_indices in radd_windows:
        for idx in epoch_indices:
            radd_counter[int(idx)] += 1
    method_counts["radd"] = radd_counter

    for method, counter in method_counts.items():
        for bin_id in sorted(set(bin_map.values())):
            indices = [int(idx) for idx in victim_train_list if bin_map[int(idx)] == bin_id]
            exposures = np.asarray([counter.get(idx, 0) for idx in indices], dtype=np.float32)
            scores = np.asarray([mem_scores[idx] for idx in indices], dtype=np.float32)
            rows.append({
                "dataset": args.dataset_name,
                "architecture": args.model_name,
                "method": method,
                "risk_bin": int(bin_id),
                "n_member": int(len(indices)),
                "memscore_mean": float(scores.mean()) if len(scores) else np.nan,
                "exposure_mean": float(exposures.mean()) if len(exposures) else np.nan,
                "exposure_max": float(exposures.max()) if len(exposures) else np.nan,
                "exposure_min": float(exposures.min()) if len(exposures) else np.nan,
                "exposure_zero_frac": float((exposures == 0).mean()) if len(exposures) else np.nan,
            })
    return rows


def main(args):
    torch.manual_seed(7)
    np.random.seed(7)
    device = "cuda:{}".format(args.device)
    cudnn.benchmark = True

    save_folder = "result/{}_{}".format(args.dataset_name, args.model_name)
    output_dir = args.output_dir or os.path.join("log", "radd_mechanism", "{}_{}".format(args.dataset_name, args.model_name))
    os.makedirs(output_dir, exist_ok=True)

    train_evalset = get_dataset(args.dataset_name, train=True, augment=False)
    testset = get_dataset(args.dataset_name, train=False)
    if testset is None:
        _ = train_evalset
    else:
        _ = ConcatDataset([train_evalset, testset])

    data_path = os.path.join(save_folder, "data_prepare.pkl")
    with open(data_path, "rb") as f:
        victim_train_list, _, _, victim_test_dataset, _, _ = pickle.load(f)

    mem_scores = load_mem_scores(args.dataset_name, args.model_name)
    bin_map = assign_risk_bins(victim_train_list, mem_scores, args.num_bins)

    teacher = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    teacher_path = os.path.join(save_folder, "victim_model", "best.pth")
    teacher.load(teacher_path)
    teacher.model.eval()

    member_loader = DataLoader(
        IndexedDataset(train_evalset, victim_train_list),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False
    )
    nonmember_loader = DataLoader(
        IndexedDataset(victim_test_dataset, None),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False
    )

    metric_rows = []
    overall_rows = []
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        model, model_dir = load_pruned_model(args, device, save_folder, method)
        if model is None:
            continue
        print("Analyze {} from {}".format(method, model_dir))
        member_df = evaluate_dataset(
            model, member_loader, device, args.num_cls,
            teacher_model=teacher.model, temp=args.distill_temp
        )
        nonmember_df = evaluate_dataset(
            model, nonmember_loader, device, args.num_cls,
            teacher_model=teacher.model, temp=args.distill_temp
        )
        metric_rows.extend(aggregate_metric_gaps(member_df, nonmember_df, bin_map, mem_scores, method, args))
        member_mean = member_df.mean(numeric_only=True)
        nonmember_mean = nonmember_df.mean(numeric_only=True)
        overall = {
            "dataset": args.dataset_name,
            "architecture": args.model_name,
            "method": method,
            "n_member": int(len(member_df)),
            "n_nonmember": int(len(nonmember_df)),
        }
        for metric in ["confidence", "entropy", "margin", "loss", "correct", "teacher_kl"]:
            overall["member_" + metric] = float(member_mean[metric])
            overall["nonmember_" + metric] = float(nonmember_mean[metric])
            overall[metric + "_gap"] = float(member_mean[metric] - nonmember_mean[metric])
        overall_rows.append(overall)

    exposure = pd.DataFrame(exposure_rows(args, victim_train_list, mem_scores, bin_map))
    metrics = pd.DataFrame(metric_rows)
    overall = pd.DataFrame(overall_rows)

    exposure_path = os.path.join(output_dir, "mechanism_exposure_bins.csv")
    metrics_path = os.path.join(output_dir, "mechanism_risk_bins.csv")
    overall_path = os.path.join(output_dir, "mechanism_overall.csv")
    exposure.to_csv(exposure_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    overall.to_csv(overall_path, index=False)

    print("Saved {}".format(exposure_path))
    print("Saved {}".format(metrics_path))
    print("Saved {}".format(overall_path))


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    main(args)
