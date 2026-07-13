import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset

from base_model import BaseModel
from datasets import get_dataset
from utils import seed_worker


parser = argparse.ArgumentParser()
parser.add_argument('device', default=0, type=int, help="GPU id to use")
parser.add_argument('config_path', default=0, type=str, help="config file path")
parser.add_argument('--dataset_name', default='cifar10', type=str)
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--optimizer', default="adam", type=str)
parser.add_argument('--k_folds', default=5, type=int)
parser.add_argument('--mem_epochs', default=30, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--output_dir', default='memscore', type=str)
parser.add_argument('--overwrite', action='store_true')


def predict_true_label_probs(model, dataset, args, device):
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    probs = np.zeros(len(dataset), dtype=np.float32)
    model.eval()
    offset = 0
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = softmax(model(inputs))
            batch_probs = outputs[torch.arange(targets.size(0)).to(device), targets]
            batch_size = targets.size(0)
            probs[offset:offset + batch_size] = batch_probs.detach().cpu().numpy()
            offset += batch_size
    return probs


def normalize_scores(raw_scores):
    raw_scores = np.asarray(raw_scores, dtype=np.float32)
    min_score = float(raw_scores.min())
    max_score = float(raw_scores.max())
    if max_score - min_score < 1e-12:
        return np.zeros_like(raw_scores, dtype=np.float32)
    return (raw_scores - min_score) / (max_score - min_score)


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = f"cuda:{args.device}"
    cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"memscore_{args.dataset_name}_{args.model_name}.csv")
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Skip existing memscore file: {output_path}")
        return

    trainset = get_dataset(args.dataset_name, train=True, augment=True)
    score_trainset = get_dataset(args.dataset_name, train=True, augment=False)
    data_size = len(trainset)
    indices = np.arange(data_size)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(indices)
    folds = [fold.astype(np.int64) for fold in np.array_split(indices, args.k_folds)]
    fold_of_index = np.zeros(data_size, dtype=np.int64)
    for fold_id, fold_indices in enumerate(folds):
        fold_of_index[fold_indices] = fold_id

    all_model_probs = []
    for fold_id, fold_indices in enumerate(folds):
        print(f"Train memscore submodel {fold_id + 1}/{args.k_folds}, subset size {len(fold_indices)}")
        subset = Subset(trainset, fold_indices.tolist())
        loader = DataLoader(
            subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker)
        sub_model = BaseModel(
            args.model_name, num_cls=args.num_cls, input_dim=args.input_dim,
            device=device, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.mem_epochs):
            sub_model.train(loader, f"MemScore Fold {fold_id} Epoch {epoch}")
        fold_probs = predict_true_label_probs(sub_model.model, score_trainset, args, device)
        all_model_probs.append(fold_probs)

    all_model_probs = np.stack(all_model_probs, axis=0)
    raw_scores = np.zeros(data_size, dtype=np.float32)
    for idx in range(data_size):
        in_fold = fold_of_index[idx]
        p_in = all_model_probs[in_fold, idx]
        if args.k_folds <= 1:
            p_out = 0.0
        else:
            p_out = (all_model_probs[:, idx].sum() - p_in) / (args.k_folds - 1)
        raw_scores[idx] = p_in - p_out

    scores = normalize_scores(raw_scores)
    pd.DataFrame(scores).to_csv(output_path)
    print(f"Saved memscore file: {output_path}")
    print(f"Raw score range: min={float(raw_scores.min()):.6f}, max={float(raw_scores.max()):.6f}")
    print(f"Normalized score range: min={float(scores.min()):.6f}, max={float(scores.max()):.6f}")


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    print(args)
    main(args)
