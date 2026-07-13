import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset, ConcatDataset
import os
import pickle
import matplotlib.pyplot as plt
import argparse
import random
from pynvml import *
from utils import *
from datasets import (
    TinyImageNetDataset,
    _convert_to_rgb,
    STL10Dataset,
    GTSRBDataset,
    Flowers102Dataset,
    FGVCAircraftDataset,
    DTDDataset,
    OxfordIIITPetDataset,
    StanfordDogsDataset,
    Food101Dataset,
    RESISC45Dataset,
    PatternNetDataset,
    PlantVillageDataset,
    OfficeHomeDataset,
    Office10Dataset,
    Kuzushiji49Dataset,
    EMNISTBalancedDataset,
    MITIndoor67Dataset,
    Caltech256BalancedDataset,
    Caltech10Dataset,
    MiniImageNetDataset,
    NICODataset,
    LOCATION_STYLE_DATASETS,
    FGVC_AIRCRAFT_IMAGE_SIZE,
    DTD_IMAGE_SIZE,
    OXFORD_PET_IMAGE_SIZE,
    STANFORD_DOGS_IMAGE_SIZE,
    FOOD101_IMAGE_SIZE,
    RESISC45_IMAGE_SIZE,
    PATTERNNET_IMAGE_SIZE,
    PLANTVILLAGE_IMAGE_SIZE,
    OFFICE_HOME_IMAGE_SIZE,
    OFFICE10_IMAGE_SIZE,
    KUZUSHIJI49_IMAGE_SIZE,
    EMNIST_BALANCED_IMAGE_SIZE,
    MIT_INDOOR67_IMAGE_SIZE,
    CALTECH256_BALANCED_IMAGE_SIZE,
    CALTECH10_IMAGE_SIZE,
    STL10_IMAGE_SIZE,
    MINIIMAGENET_IMAGE_SIZE,
    NICO_IMAGE_SIZE,
)


parser = argparse.ArgumentParser()

data_score_path = 'memscore'
LOCATION_STYLE_NUM_CLASSES = {
    "location": 30,
    "foursquare_nyc": 30,
    "foursquare_tky": 30,
    "brightkite": 30,
    "gowalla": 30,
}


def is_location_style_dataset(dataset_name):
    return dataset_name in LOCATION_STYLE_DATASETS


def load_location_style_trainset(dataset_name):
    pkl_path = LOCATION_STYLE_DATASETS[dataset_name][0]
    with open(pkl_path, 'rb') as f:
        trainset, _ = pickle.load(f)
    return trainset


def get_location_style_num_classes(dataset_name):
    return LOCATION_STYLE_NUM_CLASSES[dataset_name]


def load_mem_scores(dataset_name, model):
    data_path = f"./{data_score_path}/memscore_{dataset_name}_{model}.csv"
    scores = pd.read_csv(data_path)
    scores = np.array(scores)
    return scores[:, 1].astype(np.float32)


def build_stl10_dataset(train=True, augment=False):
    mean = (0.4467, 0.4398, 0.4066)
    std = (0.2603, 0.2566, 0.2713)
    if train and augment:
        transform = transforms.Compose([
            transforms.RandomCrop(STL10_IMAGE_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((STL10_IMAGE_SIZE, STL10_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return STL10Dataset(root='./data/datasets/stl10-data', train=train, transform=transform)


class MemScoreSubset(Dataset):
    def __init__(self, dataset, indices, mem_scores):
        self.dataset = dataset
        self.indices = list(indices)
        self.mem_scores = mem_scores

    def __getitem__(self, index):
        data_index = self.indices[index]
        inputs, targets = self.dataset[data_index]
        return inputs, targets, float(self.mem_scores[data_index])

    def __len__(self):
        return len(self.indices)


def get_memscore_dataset(dataset_name):
    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data/datasets/cifar10-data', train=True, download=False, transform=transform)
    elif dataset_name == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR100(root='./data/datasets/cifar100-data', train=True, download=False, transform=transform)
    elif dataset_name == "tinyimagenet":
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tiny_size = int(os.environ.get('TINYIMAGENET_SIZE', '64'))
        dataset = TinyImageNetDataset(root='./data/datasets/tinyimagenet', train=True, transform=transform, size=tiny_size)
    elif dataset_name == "cinic":
        with open("./data/datasets/cinic/cinic.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "texas100":
        with open("./data/datasets/texas/texas100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "purchase100":
        with open("./data/datasets/purchase100/purchase100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif is_location_style_dataset(dataset_name):
        dataset = load_location_style_trainset(dataset_name)
    elif dataset_name == "gtsrb":
        mean = (0.3403, 0.3121, 0.3214)
        std = (0.2724, 0.2608, 0.2669)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = GTSRBDataset(root='./data/datasets/gtsrb', train=True, transform=transform)
    elif dataset_name == "flowers102":
        with open('./data/datasets/flowers102_split.pkl', 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "caltech101":
        with open("./data/datasets/caltech101/caltech101.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "caltech10":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((CALTECH10_IMAGE_SIZE, CALTECH10_IMAGE_SIZE)),
            transforms.Lambda(_convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Caltech10Dataset(root='./data/datasets/caltech10', train=True, transform=transform)
    elif dataset_name == "fgvc_aircraft":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((FGVC_AIRCRAFT_IMAGE_SIZE, FGVC_AIRCRAFT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = FGVCAircraftDataset(root='./data/datasets/fgvc_aircraft', train=True, transform=transform)
    elif dataset_name == "dtd":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((DTD_IMAGE_SIZE, DTD_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = DTDDataset(root='./data/datasets/dtd', train=True, transform=transform)
    elif dataset_name == "oxford_pet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((OXFORD_PET_IMAGE_SIZE, OXFORD_PET_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = OxfordIIITPetDataset(root='./data/datasets/oxford_pet', train=True, transform=transform)
    elif dataset_name == "stanford_dogs":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((STANFORD_DOGS_IMAGE_SIZE, STANFORD_DOGS_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = StanfordDogsDataset(root='./data/datasets/stanford_dogs', train=True, transform=transform)
    elif dataset_name == "food101":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((FOOD101_IMAGE_SIZE, FOOD101_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Food101Dataset(root='./data/datasets/food101', train=True, transform=transform)
    elif dataset_name == "resisc45":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((RESISC45_IMAGE_SIZE, RESISC45_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = RESISC45Dataset(root='./data/datasets/resisc45', train=True, transform=transform)
    elif dataset_name == "patternnet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((PATTERNNET_IMAGE_SIZE, PATTERNNET_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = PatternNetDataset(root='./data/datasets/patternnet', train=True, transform=transform)
    elif dataset_name == "plantvillage":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((PLANTVILLAGE_IMAGE_SIZE, PLANTVILLAGE_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = PlantVillageDataset(root='./data/datasets/plantvillage', train=True, transform=transform)
    elif dataset_name == "office_home":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((OFFICE_HOME_IMAGE_SIZE, OFFICE_HOME_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = OfficeHomeDataset(root='./data/datasets/office_home', train=True, transform=transform)
    elif dataset_name == "office10":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((OFFICE10_IMAGE_SIZE, OFFICE10_IMAGE_SIZE)),
            transforms.Lambda(_convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Office10Dataset(root='./data/datasets/office10', train=True, transform=transform)
    elif dataset_name == "kuzushiji49":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform = transforms.Compose([
            transforms.Resize((KUZUSHIJI49_IMAGE_SIZE, KUZUSHIJI49_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Kuzushiji49Dataset(root='./data/datasets/kuzushiji49', train=True, transform=transform)
    elif dataset_name == "emnist_balanced":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform = transforms.Compose([
            transforms.Resize((EMNIST_BALANCED_IMAGE_SIZE, EMNIST_BALANCED_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = EMNISTBalancedDataset(root='./data/datasets/emnist_balanced', train=True, transform=transform)
    elif dataset_name == "mit_indoor67":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((MIT_INDOOR67_IMAGE_SIZE, MIT_INDOOR67_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = MITIndoor67Dataset(root='./data/datasets/mit_indoor67', train=True, transform=transform)
    elif dataset_name == "caltech256_balanced":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((CALTECH256_BALANCED_IMAGE_SIZE, CALTECH256_BALANCED_IMAGE_SIZE)),
            transforms.Lambda(_convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Caltech256BalancedDataset(root='./data/datasets/caltech256_balanced', train=True, transform=transform)
    elif dataset_name == "miniimagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((MINIIMAGENET_IMAGE_SIZE, MINIIMAGENET_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = MiniImageNetDataset(root='./data/datasets/miniimagenet', train=True, transform=transform)
    elif dataset_name == "nico":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((NICO_IMAGE_SIZE, NICO_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = NICODataset(root='./data/datasets/nico', train=True, transform=transform)
    elif dataset_name == "stl10":
        dataset = build_stl10_dataset(train=True)
    else:
        raise ValueError(dataset_name)
    return dataset


def get_dataset_num_classes(dataset_name, dataset):
    if dataset_name == "texas100" or dataset_name == "purchase100":
        return 100
    elif dataset_name == "tinyimagenet":
        return 200
    elif is_location_style_dataset(dataset_name):
        return get_location_style_num_classes(dataset_name)
    elif dataset_name == "cinic":
        return 10
    elif dataset_name == "stl10":
        return 10
    elif dataset_name == "gtsrb":
        return 43
    elif dataset_name == "flowers102":
        return 102
    elif dataset_name == "caltech101":
        return 101
    elif dataset_name == "caltech10":
        return 10
    elif dataset_name == "fgvc_aircraft":
        return 100
    elif dataset_name == "dtd":
        return 47
    elif dataset_name == "oxford_pet":
        return 37
    elif dataset_name == "stanford_dogs":
        return 120
    elif dataset_name == "food101":
        return 101
    elif dataset_name == "resisc45":
        return 45
    elif dataset_name == "patternnet":
        return 38
    elif dataset_name == "plantvillage":
        return 38
    elif dataset_name == "office_home":
        return 65
    elif dataset_name == "office10":
        return 10
    elif dataset_name == "kuzushiji49":
        return 49
    elif dataset_name == "emnist_balanced":
        return 47
    elif dataset_name == "mit_indoor67":
        return 67
    elif dataset_name == "caltech256_balanced":
        return 100
    elif dataset_name == "miniimagenet":
        return 100
    elif dataset_name == "nico":
        return 19
    return len(dataset.classes)


def risk_aware_window(dataset_name, model, total_epochs, train_list, width, stride, risk_gamma=1.0):
    mem_scores = load_mem_scores(dataset_name, model)
    dataset = get_memscore_dataset(dataset_name)
    num_classes = get_dataset_num_classes(dataset_name, dataset)
    class_indices = [[] for _ in range(num_classes)]

    for idx in train_list:
        _, target = dataset[idx]
        class_indices[target].append(idx)

    exposures = dict((idx, 0) for idx in train_list)
    epochs_data_idx = []
    exposure_strength = max(1.0, np.log1p(stride if stride is not None else 1.0))

    for _ in range(total_epochs):
        data_idx = []
        for classlist in class_indices:
            if len(classlist) == 0:
                continue
            candidates = np.array(classlist)
            risks = np.clip(mem_scores[candidates], 0.0, 1.0)
            reuse = np.array([exposures[int(idx)] for idx in candidates], dtype=np.float32)
            privacy_weight = np.power(1.0 - risks, risk_gamma) + 0.05
            sample_weight = privacy_weight / (1.0 + exposure_strength * reuse)
            if sample_weight.sum() <= 0:
                sample_weight = np.ones_like(sample_weight)
            sample_weight = sample_weight / sample_weight.sum()
            sample_size = len(candidates) if width is None or width <= 0 else min(width, len(candidates))
            chosen = np.random.choice(candidates, size=sample_size, replace=False, p=sample_weight)
            data_idx.extend(chosen.tolist())
            for idx in chosen:
                exposures[int(idx)] += 1
        epochs_data_idx.append(data_idx)

    return epochs_data_idx


def _take_rotating(items, start, count):
    if count <= 0 or len(items) == 0:
        return []
    count = min(count, len(items))
    start = start % len(items)
    end = start + count
    if end <= len(items):
        return items[start:end]
    return items[start:] + items[:end - len(items)]


def risk_budget_window(dataset_name, model, total_epochs, train_list, width, stride,
                       mem_thre, high_risk_cap=1, high_risk_ratio=0.1):
    mem_scores = load_mem_scores(dataset_name, model)
    dataset = get_memscore_dataset(dataset_name)
    num_classes = get_dataset_num_classes(dataset_name, dataset)
    class_indices = [[] for _ in range(num_classes)]

    for idx in train_list:
        _, target = dataset[idx]
        class_indices[target].append(idx)

    exposures = dict((idx, 0) for idx in train_list)
    epochs_data_idx = []
    high_risk_cap = max(0, int(high_risk_cap))
    high_risk_ratio = max(0.0, min(1.0, high_risk_ratio))
    stride = 1 if stride is None or stride <= 0 else stride

    for epoch in range(total_epochs):
        data_idx = []
        for classlist in class_indices:
            if len(classlist) == 0:
                continue

            class_width = len(classlist) if width is None or width <= 0 else min(width, len(classlist))
            sorted_items = sorted(classlist, key=lambda idx: mem_scores[idx])
            low_items = [idx for idx in sorted_items if mem_scores[idx] < mem_thre]
            high_items = [idx for idx in sorted_items if mem_scores[idx] >= mem_thre and exposures[idx] < high_risk_cap]

            high_quota = min(int(round(class_width * high_risk_ratio)), len(high_items))
            low_quota = max(0, class_width - high_quota)

            selected_low = _take_rotating(low_items, stride * epoch, low_quota)
            selected_high = _take_rotating(high_items, stride * epoch, high_quota)
            selected = selected_low + selected_high

            if len(selected) < class_width:
                selected_set = set(selected)
                refill_items = [
                    idx for idx in sorted_items
                    if idx not in selected_set and (mem_scores[idx] < mem_thre or exposures[idx] < high_risk_cap)
                ]
                selected.extend(_take_rotating(refill_items, stride * epoch, class_width - len(selected)))

            data_idx.extend(selected)
            for idx in selected:
                exposures[idx] += 1
        epochs_data_idx.append(data_idx)

    return epochs_data_idx

# Data score order: from high to low (RSW H->L / SWMR H->L)
def slide_window(dataset_name, model, total_epochs, train_list, width, stride):
    data_path = f"./{data_score_path}/memscore_{dataset_name}_{model}.csv"
    s=pd.read_csv(data_path)
    s=np.array(s)

    if dataset_name in ("miniimagenet", "nico", "caltech10", "office10", "kuzushiji49", "emnist_balanced"):
        dataset = get_memscore_dataset(dataset_name)
    elif dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data/datasets/cifar10-data', train=True, download=False, transform=transform)
    elif dataset_name == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR100(root='./data/datasets/cifar100-data', train=True, download=False, transform=transform)
    elif dataset_name == "tinyimagenet":
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tiny_size = int(os.environ.get('TINYIMAGENET_SIZE', '64'))
        dataset = TinyImageNetDataset(root='./data/datasets/tinyimagenet', train=True, transform=transform, size=tiny_size)
    elif dataset_name == "cinic":
        with open("./data/datasets/cinic/cinic.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "texas100":
        with open("./data/datasets/texas/texas100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "purchase100":
        with open("./data/datasets/purchase100/purchase100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif is_location_style_dataset(dataset_name):
        dataset = load_location_style_trainset(dataset_name)
    elif dataset_name == "gtsrb":
        mean = (0.3403, 0.3121, 0.3214)
        std = (0.2724, 0.2608, 0.2669)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = GTSRBDataset(root='./data/datasets/gtsrb', train=True, transform=transform)
    elif dataset_name == "flowers102":
        with open('./data/datasets/flowers102_split.pkl', 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "caltech101":
        with open("./data/datasets/caltech101/caltech101.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "fgvc_aircraft":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((FGVC_AIRCRAFT_IMAGE_SIZE, FGVC_AIRCRAFT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = FGVCAircraftDataset(root='./data/datasets/fgvc_aircraft', train=True, transform=transform)
    elif dataset_name == "dtd":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((DTD_IMAGE_SIZE, DTD_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = DTDDataset(root='./data/datasets/dtd', train=True, transform=transform)
    elif dataset_name == "oxford_pet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((OXFORD_PET_IMAGE_SIZE, OXFORD_PET_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = OxfordIIITPetDataset(root='./data/datasets/oxford_pet', train=True, transform=transform)
    elif dataset_name == "stanford_dogs":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((STANFORD_DOGS_IMAGE_SIZE, STANFORD_DOGS_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = StanfordDogsDataset(root='./data/datasets/stanford_dogs', train=True, transform=transform)
    elif dataset_name == "food101":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((FOOD101_IMAGE_SIZE, FOOD101_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Food101Dataset(root='./data/datasets/food101', train=True, transform=transform)
    elif dataset_name == "resisc45":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((RESISC45_IMAGE_SIZE, RESISC45_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = RESISC45Dataset(root='./data/datasets/resisc45', train=True, transform=transform)
    elif dataset_name == "patternnet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((PATTERNNET_IMAGE_SIZE, PATTERNNET_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = PatternNetDataset(root='./data/datasets/patternnet', train=True, transform=transform)
    elif dataset_name == "plantvillage":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((PLANTVILLAGE_IMAGE_SIZE, PLANTVILLAGE_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = PlantVillageDataset(root='./data/datasets/plantvillage', train=True, transform=transform)
    elif dataset_name == "office_home":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((OFFICE_HOME_IMAGE_SIZE, OFFICE_HOME_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = OfficeHomeDataset(root='./data/datasets/office_home', train=True, transform=transform)
    elif dataset_name == "mit_indoor67":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((MIT_INDOOR67_IMAGE_SIZE, MIT_INDOOR67_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = MITIndoor67Dataset(root='./data/datasets/mit_indoor67', train=True, transform=transform)
    elif dataset_name == "caltech256_balanced":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((CALTECH256_BALANCED_IMAGE_SIZE, CALTECH256_BALANCED_IMAGE_SIZE)),
            transforms.Lambda(_convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Caltech256BalancedDataset(root='./data/datasets/caltech256_balanced', train=True, transform=transform)
    elif dataset_name == "stl10":
        dataset = build_stl10_dataset(train=True)

    if dataset_name == "texas100" or dataset_name == "purchase100":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    elif is_location_style_dataset(dataset_name):
        class_indices = [[] for _ in range(get_location_style_num_classes(dataset_name))]
        class_score_indices = [[] for _ in range(get_location_style_num_classes(dataset_name))]
    elif dataset_name == "cinic":
        class_indices = [[] for _ in range(10)]
        class_score_indices = [[] for _ in range(10)]
    elif dataset_name == "stl10":
        class_indices = [[] for _ in range(10)]
        class_score_indices = [[] for _ in range(10)]
    elif dataset_name == "gtsrb":
        class_indices = [[] for _ in range(43)]
        class_score_indices = [[] for _ in range(43)]
    elif dataset_name == "flowers102":
        class_indices = [[] for _ in range(102)]
        class_score_indices = [[] for _ in range(102)]
    elif dataset_name == "caltech101":
        class_indices = [[] for _ in range(101)]
        class_score_indices = [[] for _ in range(101)]
    elif dataset_name == "fgvc_aircraft":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    elif dataset_name == "dtd":
        class_indices = [[] for _ in range(47)]
        class_score_indices = [[] for _ in range(47)]
    elif dataset_name == "oxford_pet":
        class_indices = [[] for _ in range(37)]
        class_score_indices = [[] for _ in range(37)]
    elif dataset_name == "stanford_dogs":
        class_indices = [[] for _ in range(120)]
        class_score_indices = [[] for _ in range(120)]
    elif dataset_name == "food101":
        class_indices = [[] for _ in range(101)]
        class_score_indices = [[] for _ in range(101)]
    elif dataset_name == "resisc45":
        class_indices = [[] for _ in range(45)]
        class_score_indices = [[] for _ in range(45)]
    elif dataset_name == "patternnet":
        class_indices = [[] for _ in range(38)]
        class_score_indices = [[] for _ in range(38)]
    elif dataset_name == "plantvillage":
        class_indices = [[] for _ in range(38)]
        class_score_indices = [[] for _ in range(38)]
    elif dataset_name == "office_home":
        class_indices = [[] for _ in range(65)]
        class_score_indices = [[] for _ in range(65)]
    elif dataset_name == "mit_indoor67":
        class_indices = [[] for _ in range(67)]
        class_score_indices = [[] for _ in range(67)]
    elif dataset_name == "caltech256_balanced":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    else:
        class_indices = [[] for _ in range(len(dataset.classes))]
        class_score_indices = [[] for _ in range(len(dataset.classes))]

    for idx in train_list:
        _, target = dataset[idx]
        class_indices[target].append(idx)
        class_score_indices[target].append(s[idx][1])

    sorted_list = []
    sorted_dict = []
    for one_class_idx, one_class_value in zip(class_indices, class_score_indices):
        class_dict = dict(zip(one_class_idx, one_class_value))
        sort_score = sorted(class_dict.items(), key=lambda x:x[1], reverse=True)
        sorted_dict.append(sort_score)
        temp_list = []
        for i in range(len(sort_score)):
            temp_list.append(sort_score[i][0])
        sorted_list.append(temp_list)                         # "sorted_list" includes 10 sublists (for example, CIFAR10 and CINIC), corresponding with 10 classes, and the elements    
                                                             # in per sublist is data's idx sorted by data mem-score 
    # slide window on dataset
    epochs_data_idx = []                                                       
    for i in range(total_epochs):
        data_idx = []
        start = stride * i
        end = start + width
        for classlist in sorted_list:
            if start >= len(classlist):
                continue
            if end > len(classlist):
                end = len(classlist)
            data_idx.extend(classlist[start:end])
        epochs_data_idx.append(data_idx)
    
    return epochs_data_idx

# Data score order reverse: from low to high  (RSW L->H / SWMR L->H)
def slide_window_reverse(dataset_name, model, total_epochs, train_list, width, stride):
    data_path = f"./{data_score_path}/memscore_{dataset_name}_{model}.csv"
    s=pd.read_csv(data_path)
    s=np.array(s)

    if dataset_name in ("miniimagenet", "nico", "caltech10", "office10", "kuzushiji49", "emnist_balanced"):
        dataset = get_memscore_dataset(dataset_name)
    elif dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data/datasets/cifar10-data', train=True, download=False, transform=transform)
    elif dataset_name == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR100(root='./data/datasets/cifar100-data', train=True, download=False, transform=transform)
    elif dataset_name == "tinyimagenet":
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tiny_size = int(os.environ.get('TINYIMAGENET_SIZE', '64'))
        dataset = TinyImageNetDataset(root='./data/datasets/tinyimagenet', train=True, transform=transform, size=tiny_size)
    elif dataset_name == "cinic":
        with open("./data/datasets/cinic/cinic.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    
    elif dataset_name == "texas100":
        with open("./data/datasets/texas/texas100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "purchase100":
        with open("./data/datasets/purchase100/purchase100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif is_location_style_dataset(dataset_name):
        dataset = load_location_style_trainset(dataset_name)
    elif dataset_name == "gtsrb":
        mean = (0.3403, 0.3121, 0.3214)
        std = (0.2724, 0.2608, 0.2669)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = GTSRBDataset(root='./data/datasets/gtsrb', train=True, transform=transform)
    elif dataset_name == "flowers102":
        with open('./data/datasets/flowers102_split.pkl', 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "caltech101":
        with open("./data/datasets/caltech101/caltech101.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "fgvc_aircraft":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((FGVC_AIRCRAFT_IMAGE_SIZE, FGVC_AIRCRAFT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = FGVCAircraftDataset(root='./data/datasets/fgvc_aircraft', train=True, transform=transform)
    elif dataset_name == "dtd":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((DTD_IMAGE_SIZE, DTD_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = DTDDataset(root='./data/datasets/dtd', train=True, transform=transform)
    elif dataset_name == "oxford_pet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((OXFORD_PET_IMAGE_SIZE, OXFORD_PET_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = OxfordIIITPetDataset(root='./data/datasets/oxford_pet', train=True, transform=transform)
    elif dataset_name == "stanford_dogs":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((STANFORD_DOGS_IMAGE_SIZE, STANFORD_DOGS_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = StanfordDogsDataset(root='./data/datasets/stanford_dogs', train=True, transform=transform)
    elif dataset_name == "food101":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((FOOD101_IMAGE_SIZE, FOOD101_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Food101Dataset(root='./data/datasets/food101', train=True, transform=transform)
    elif dataset_name == "resisc45":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((RESISC45_IMAGE_SIZE, RESISC45_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = RESISC45Dataset(root='./data/datasets/resisc45', train=True, transform=transform)
    elif dataset_name == "patternnet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((PATTERNNET_IMAGE_SIZE, PATTERNNET_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = PatternNetDataset(root='./data/datasets/patternnet', train=True, transform=transform)
    elif dataset_name == "plantvillage":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((PLANTVILLAGE_IMAGE_SIZE, PLANTVILLAGE_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = PlantVillageDataset(root='./data/datasets/plantvillage', train=True, transform=transform)
    elif dataset_name == "office_home":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((OFFICE_HOME_IMAGE_SIZE, OFFICE_HOME_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = OfficeHomeDataset(root='./data/datasets/office_home', train=True, transform=transform)
    elif dataset_name == "mit_indoor67":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((MIT_INDOOR67_IMAGE_SIZE, MIT_INDOOR67_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = MITIndoor67Dataset(root='./data/datasets/mit_indoor67', train=True, transform=transform)
    elif dataset_name == "caltech256_balanced":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((CALTECH256_BALANCED_IMAGE_SIZE, CALTECH256_BALANCED_IMAGE_SIZE)),
            transforms.Lambda(_convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Caltech256BalancedDataset(root='./data/datasets/caltech256_balanced', train=True, transform=transform)
    elif dataset_name == "stl10":
        dataset = build_stl10_dataset(train=True)
    if dataset_name == "texas100" or dataset_name == "purchase100":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    elif is_location_style_dataset(dataset_name):
        class_indices = [[] for _ in range(get_location_style_num_classes(dataset_name))]
        class_score_indices = [[] for _ in range(get_location_style_num_classes(dataset_name))]
    elif dataset_name == "cinic":
        class_indices = [[] for _ in range(10)]
        class_score_indices = [[] for _ in range(10)]
    elif dataset_name == "stl10":
        class_indices = [[] for _ in range(10)]
        class_score_indices = [[] for _ in range(10)]
    elif dataset_name == "gtsrb":
        class_indices = [[] for _ in range(43)]
        class_score_indices = [[] for _ in range(43)]
    elif dataset_name == "flowers102":
        class_indices = [[] for _ in range(102)]
        class_score_indices = [[] for _ in range(102)]
    elif dataset_name == "caltech101":
        class_indices = [[] for _ in range(101)]
        class_score_indices = [[] for _ in range(101)]
    elif dataset_name == "fgvc_aircraft":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    elif dataset_name == "dtd":
        class_indices = [[] for _ in range(47)]
        class_score_indices = [[] for _ in range(47)]
    elif dataset_name == "oxford_pet":
        class_indices = [[] for _ in range(37)]
        class_score_indices = [[] for _ in range(37)]
    elif dataset_name == "stanford_dogs":
        class_indices = [[] for _ in range(120)]
        class_score_indices = [[] for _ in range(120)]
    elif dataset_name == "food101":
        class_indices = [[] for _ in range(101)]
        class_score_indices = [[] for _ in range(101)]
    elif dataset_name == "resisc45":
        class_indices = [[] for _ in range(45)]
        class_score_indices = [[] for _ in range(45)]
    elif dataset_name == "patternnet":
        class_indices = [[] for _ in range(38)]
        class_score_indices = [[] for _ in range(38)]
    elif dataset_name == "plantvillage":
        class_indices = [[] for _ in range(38)]
        class_score_indices = [[] for _ in range(38)]
    elif dataset_name == "office_home":
        class_indices = [[] for _ in range(65)]
        class_score_indices = [[] for _ in range(65)]
    elif dataset_name == "mit_indoor67":
        class_indices = [[] for _ in range(67)]
        class_score_indices = [[] for _ in range(67)]
    elif dataset_name == "caltech256_balanced":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    else:
        class_indices = [[] for _ in range(len(dataset.classes))]
        class_score_indices = [[] for _ in range(len(dataset.classes))]
    
    for idx in train_list:
        _, target = dataset[idx]
        class_indices[target].append(idx)
        class_score_indices[target].append(s[idx][1])

    sorted_list = []
    sorted_dict = []
    for one_class_idx, one_class_value in zip(class_indices, class_score_indices):
        class_dict = dict(zip(one_class_idx, one_class_value))
        sort_score = sorted(class_dict.items(), key=lambda x:x[1], reverse=False)
        sorted_dict.append(sort_score)
        temp_list = []
        for i in range(len(sort_score)):
            temp_list.append(sort_score[i][0])
        sorted_list.append(temp_list)                         # "sorted_list" includes 10 sublists (for example, CIFAR10 and CINIC), corresponding with 10 classes, and the elements    
                                                             # in per sublist is data's idx sorted by data mem-score 
    # slide window on dataset
    epochs_data_idx = []                                                       
    for i in range(total_epochs):
        data_idx = []
        start = stride * i
        end = start + width
        for classlist in sorted_list:
            if start >= len(classlist):
                continue
            if end > len(classlist):
                end = len(classlist)
            data_idx.extend(classlist[start:end])
        epochs_data_idx.append(data_idx)
    
    return epochs_data_idx

# Prepare data for risky memory regularization
def ml2_process(dataset_name, model, train_list, mem_thre):
    data_path = f"./{data_score_path}/memscore_{dataset_name}_{model}.csv"
    s=pd.read_csv(data_path)
    s=np.array(s)

    if dataset_name in ("miniimagenet", "nico", "caltech10", "office10", "kuzushiji49", "emnist_balanced"):
        dataset = get_memscore_dataset(dataset_name)
    elif dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data/datasets/cifar10-data', train=True, download=False, transform=transform)
    elif dataset_name == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR100(root='./data/datasets/cifar100-data', train=True, download=False, transform=transform)
    elif dataset_name == "tinyimagenet":
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tiny_size = int(os.environ.get('TINYIMAGENET_SIZE', '64'))
        dataset = TinyImageNetDataset(root='./data/datasets/tinyimagenet', train=True, transform=transform, size=tiny_size)
    elif dataset_name == "cinic":
        with open("./data/datasets/cinic/cinic.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "texas100":
        with open("./data/datasets/texas/texas100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "purchase100":
        with open("./data/datasets/purchase100/purchase100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif is_location_style_dataset(dataset_name):
        dataset = load_location_style_trainset(dataset_name)
    elif dataset_name == "gtsrb":
        mean = (0.3403, 0.3121, 0.3214)
        std = (0.2724, 0.2608, 0.2669)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = GTSRBDataset(root='./data/datasets/gtsrb', train=True, transform=transform)
    elif dataset_name == "flowers102":
        with open('./data/datasets/flowers102_split.pkl', 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "caltech101":
        with open("./data/datasets/caltech101/caltech101.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "fgvc_aircraft":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((FGVC_AIRCRAFT_IMAGE_SIZE, FGVC_AIRCRAFT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = FGVCAircraftDataset(root='./data/datasets/fgvc_aircraft', train=True, transform=transform)
    elif dataset_name == "dtd":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((DTD_IMAGE_SIZE, DTD_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = DTDDataset(root='./data/datasets/dtd', train=True, transform=transform)
    elif dataset_name == "oxford_pet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((OXFORD_PET_IMAGE_SIZE, OXFORD_PET_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = OxfordIIITPetDataset(root='./data/datasets/oxford_pet', train=True, transform=transform)
    elif dataset_name == "stanford_dogs":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((STANFORD_DOGS_IMAGE_SIZE, STANFORD_DOGS_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = StanfordDogsDataset(root='./data/datasets/stanford_dogs', train=True, transform=transform)
    elif dataset_name == "food101":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((FOOD101_IMAGE_SIZE, FOOD101_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Food101Dataset(root='./data/datasets/food101', train=True, transform=transform)
    elif dataset_name == "resisc45":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((RESISC45_IMAGE_SIZE, RESISC45_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = RESISC45Dataset(root='./data/datasets/resisc45', train=True, transform=transform)
    elif dataset_name == "patternnet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((PATTERNNET_IMAGE_SIZE, PATTERNNET_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = PatternNetDataset(root='./data/datasets/patternnet', train=True, transform=transform)
    elif dataset_name == "plantvillage":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((PLANTVILLAGE_IMAGE_SIZE, PLANTVILLAGE_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = PlantVillageDataset(root='./data/datasets/plantvillage', train=True, transform=transform)
    elif dataset_name == "office_home":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((OFFICE_HOME_IMAGE_SIZE, OFFICE_HOME_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = OfficeHomeDataset(root='./data/datasets/office_home', train=True, transform=transform)
    elif dataset_name == "mit_indoor67":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((MIT_INDOOR67_IMAGE_SIZE, MIT_INDOOR67_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = MITIndoor67Dataset(root='./data/datasets/mit_indoor67', train=True, transform=transform)
    elif dataset_name == "caltech256_balanced":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((CALTECH256_BALANCED_IMAGE_SIZE, CALTECH256_BALANCED_IMAGE_SIZE)),
            transforms.Lambda(_convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Caltech256BalancedDataset(root='./data/datasets/caltech256_balanced', train=True, transform=transform)
    elif dataset_name == "stl10":
        dataset = build_stl10_dataset(train=True)

    if dataset_name == "texas100" or dataset_name == "purchase100":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    elif is_location_style_dataset(dataset_name):
        class_indices = [[] for _ in range(get_location_style_num_classes(dataset_name))]
        class_score_indices = [[] for _ in range(get_location_style_num_classes(dataset_name))]
    elif dataset_name == "cinic":
        class_indices = [[] for _ in range(10)]
        class_score_indices = [[] for _ in range(10)]
    elif dataset_name == "stl10":
        class_indices = [[] for _ in range(10)]
        class_score_indices = [[] for _ in range(10)]
    elif dataset_name == "gtsrb":
        class_indices = [[] for _ in range(43)]
        class_score_indices = [[] for _ in range(43)]
    elif dataset_name == "flowers102":
        class_indices = [[] for _ in range(102)]
        class_score_indices = [[] for _ in range(102)]
    elif dataset_name == "caltech101":
        class_indices = [[] for _ in range(101)]
        class_score_indices = [[] for _ in range(101)]
    elif dataset_name == "fgvc_aircraft":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    elif dataset_name == "dtd":
        class_indices = [[] for _ in range(47)]
        class_score_indices = [[] for _ in range(47)]
    elif dataset_name == "oxford_pet":
        class_indices = [[] for _ in range(37)]
        class_score_indices = [[] for _ in range(37)]
    elif dataset_name == "stanford_dogs":
        class_indices = [[] for _ in range(120)]
        class_score_indices = [[] for _ in range(120)]
    elif dataset_name == "food101":
        class_indices = [[] for _ in range(101)]
        class_score_indices = [[] for _ in range(101)]
    elif dataset_name == "resisc45":
        class_indices = [[] for _ in range(45)]
        class_score_indices = [[] for _ in range(45)]
    elif dataset_name == "patternnet":
        class_indices = [[] for _ in range(38)]
        class_score_indices = [[] for _ in range(38)]
    elif dataset_name == "plantvillage":
        class_indices = [[] for _ in range(38)]
        class_score_indices = [[] for _ in range(38)]
    elif dataset_name == "office_home":
        class_indices = [[] for _ in range(65)]
        class_score_indices = [[] for _ in range(65)]
    elif dataset_name == "mit_indoor67":
        class_indices = [[] for _ in range(67)]
        class_score_indices = [[] for _ in range(67)]
    elif dataset_name == "caltech256_balanced":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    else:
        class_indices = [[] for _ in range(len(dataset.classes))]
        class_score_indices = [[] for _ in range(len(dataset.classes))]
    
    for idx in train_list:
        _, target = dataset[idx]
        class_indices[target].append(idx)
        class_score_indices[target].append(s[idx][1])

    risk_idx = []
    gen_idx = []
    for one_class_idx, one_class_value in zip(class_indices, class_score_indices):
        for i, score in enumerate(one_class_value):
            if score >= mem_thre:
                risk_idx.append(one_class_idx[i])
            else:
                gen_idx.append(one_class_idx[i])
    return risk_idx, gen_idx

# random order
def slide_random(dataset_name, total_epochs, train_list, width, stride):
    if dataset_name in ("miniimagenet", "nico", "caltech10", "office10", "kuzushiji49", "emnist_balanced"):
        dataset = get_memscore_dataset(dataset_name)
    elif dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data/datasets/cifar10-data', train=True, download=False, transform=transform)
    elif dataset_name == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR100(root='./data/datasets/cifar100-data', train=True, download=False, transform=transform)
    elif dataset_name == "tinyimagenet":
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tiny_size = int(os.environ.get('TINYIMAGENET_SIZE', '64'))
        dataset = TinyImageNetDataset(root='./data/datasets/tinyimagenet', train=True, transform=transform, size=tiny_size)
    elif dataset_name == "cinic":
        with open("./data/datasets/cinic/cinic.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "texas100":
        with open("./data/datasets/texas/texas100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "purchase100":
        with open("./data/datasets/purchase100/purchase100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif is_location_style_dataset(dataset_name):
        dataset = load_location_style_trainset(dataset_name)
    elif dataset_name == "gtsrb":
        mean = (0.3403, 0.3121, 0.3214)
        std = (0.2724, 0.2608, 0.2669)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = GTSRBDataset(root='./data/datasets/gtsrb', train=True, transform=transform)
    elif dataset_name == "flowers102":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Flowers102Dataset(root='./data/datasets', split='train', transform=transform)
    elif dataset_name == "caltech101":
        with open("./data/datasets/caltech101/caltech101.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "fgvc_aircraft":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((FGVC_AIRCRAFT_IMAGE_SIZE, FGVC_AIRCRAFT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = FGVCAircraftDataset(root='./data/datasets/fgvc_aircraft', train=True, transform=transform)
    elif dataset_name == "dtd":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((DTD_IMAGE_SIZE, DTD_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = DTDDataset(root='./data/datasets/dtd', train=True, transform=transform)
    elif dataset_name == "oxford_pet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((OXFORD_PET_IMAGE_SIZE, OXFORD_PET_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = OxfordIIITPetDataset(root='./data/datasets/oxford_pet', train=True, transform=transform)
    elif dataset_name == "stanford_dogs":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((STANFORD_DOGS_IMAGE_SIZE, STANFORD_DOGS_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = StanfordDogsDataset(root='./data/datasets/stanford_dogs', train=True, transform=transform)
    elif dataset_name == "food101":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((FOOD101_IMAGE_SIZE, FOOD101_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Food101Dataset(root='./data/datasets/food101', train=True, transform=transform)
    elif dataset_name == "resisc45":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((RESISC45_IMAGE_SIZE, RESISC45_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = RESISC45Dataset(root='./data/datasets/resisc45', train=True, transform=transform)
    elif dataset_name == "patternnet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((PATTERNNET_IMAGE_SIZE, PATTERNNET_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = PatternNetDataset(root='./data/datasets/patternnet', train=True, transform=transform)
    elif dataset_name == "plantvillage":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((PLANTVILLAGE_IMAGE_SIZE, PLANTVILLAGE_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = PlantVillageDataset(root='./data/datasets/plantvillage', train=True, transform=transform)
    elif dataset_name == "office_home":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((OFFICE_HOME_IMAGE_SIZE, OFFICE_HOME_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = OfficeHomeDataset(root='./data/datasets/office_home', train=True, transform=transform)
    elif dataset_name == "mit_indoor67":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((MIT_INDOOR67_IMAGE_SIZE, MIT_INDOOR67_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = MITIndoor67Dataset(root='./data/datasets/mit_indoor67', train=True, transform=transform)
    elif dataset_name == "caltech256_balanced":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((CALTECH256_BALANCED_IMAGE_SIZE, CALTECH256_BALANCED_IMAGE_SIZE)),
            transforms.Lambda(_convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Caltech256BalancedDataset(root='./data/datasets/caltech256_balanced', train=True, transform=transform)
    elif dataset_name == "stl10":
        dataset = build_stl10_dataset(train=True)

    if dataset_name == "texas100" or dataset_name == "purchase100":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    elif is_location_style_dataset(dataset_name):
        class_indices = [[] for _ in range(get_location_style_num_classes(dataset_name))]
        class_score_indices = [[] for _ in range(get_location_style_num_classes(dataset_name))]
    elif dataset_name == "cinic":
        class_indices = [[] for _ in range(10)]
        class_score_indices = [[] for _ in range(10)]
    elif dataset_name == "stl10":
        class_indices = [[] for _ in range(10)]
        class_score_indices = [[] for _ in range(10)]
    elif dataset_name == "gtsrb":
        class_indices = [[] for _ in range(43)]
        class_score_indices = [[] for _ in range(43)]
    elif dataset_name == "flowers102":
        class_indices = [[] for _ in range(102)]
        class_score_indices = [[] for _ in range(102)]
    elif dataset_name == "caltech101":
        class_indices = [[] for _ in range(101)]
        class_score_indices = [[] for _ in range(101)]
    elif dataset_name == "fgvc_aircraft":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    elif dataset_name == "dtd":
        class_indices = [[] for _ in range(47)]
        class_score_indices = [[] for _ in range(47)]
    elif dataset_name == "oxford_pet":
        class_indices = [[] for _ in range(37)]
        class_score_indices = [[] for _ in range(37)]
    elif dataset_name == "stanford_dogs":
        class_indices = [[] for _ in range(120)]
        class_score_indices = [[] for _ in range(120)]
    elif dataset_name == "food101":
        class_indices = [[] for _ in range(101)]
        class_score_indices = [[] for _ in range(101)]
    elif dataset_name == "resisc45":
        class_indices = [[] for _ in range(45)]
        class_score_indices = [[] for _ in range(45)]
    elif dataset_name == "patternnet":
        class_indices = [[] for _ in range(38)]
        class_score_indices = [[] for _ in range(38)]
    elif dataset_name == "plantvillage":
        class_indices = [[] for _ in range(38)]
        class_score_indices = [[] for _ in range(38)]
    elif dataset_name == "office_home":
        class_indices = [[] for _ in range(65)]
        class_score_indices = [[] for _ in range(65)]
    elif dataset_name == "mit_indoor67":
        class_indices = [[] for _ in range(67)]
        class_score_indices = [[] for _ in range(67)]
    elif dataset_name == "caltech256_balanced":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    else:
        class_indices = [[] for _ in range(len(dataset.classes))]
        class_score_indices = [[] for _ in range(len(dataset.classes))]
    
    for idx in train_list:
        _, target = dataset[idx]
        class_indices[target].append(idx)
 
    # Slide window on dataset: data mem-score is ordered randomly
    epochs_data_idx = []                                                       
    for i in range(total_epochs):
        data_idx = []
        start = stride * i
        end = start + width
        for classlist in class_indices:
            if start >= len(classlist):
                continue
            if end > len(classlist):
                end = len(classlist)
            data_idx.extend(classlist[start:end])
        epochs_data_idx.append(data_idx)
    return epochs_data_idx
        
