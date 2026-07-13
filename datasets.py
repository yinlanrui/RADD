import os
import pickle
import gzip
import struct
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, Subset
from PIL import Image

try:
    from scipy.io import loadmat as _loadmat
except ImportError:
    _loadmat = None


FGVC_AIRCRAFT_IMAGE_SIZE = 64
DTD_IMAGE_SIZE = 64
OXFORD_PET_IMAGE_SIZE = 64
STANFORD_DOGS_IMAGE_SIZE = 64
FOOD101_IMAGE_SIZE = 64
RESISC45_IMAGE_SIZE = 64
PATTERNNET_IMAGE_SIZE = 64
PLANTVILLAGE_IMAGE_SIZE = 64
OFFICE_HOME_IMAGE_SIZE = 64
OFFICE10_IMAGE_SIZE = 64
MIT_INDOOR67_IMAGE_SIZE = 96
CALTECH256_BALANCED_IMAGE_SIZE = 96
CALTECH10_IMAGE_SIZE = 96
STL10_IMAGE_SIZE = 96
MINIIMAGENET_IMAGE_SIZE = 84
NICO_IMAGE_SIZE = 64
KUZUSHIJI49_IMAGE_SIZE = 32
EMNIST_BALANCED_IMAGE_SIZE = 32
LOCATION_STYLE_DATASETS = {
    "location": ("./data/datasets/location/location.pkl", "./data/datasets/location/data_complete.npz", True),
    "foursquare_nyc": ("./data/datasets/foursquare_nyc/foursquare_nyc.pkl", "./data/datasets/foursquare_nyc/data_complete.npz", False),
    "foursquare_tky": ("./data/datasets/foursquare_tky/foursquare_tky.pkl", "./data/datasets/foursquare_tky/data_complete.npz", False),
    "brightkite": ("./data/datasets/brightkite/brightkite.pkl", "./data/datasets/brightkite/data_complete.npz", False),
    "gowalla": ("./data/datasets/gowalla/gowalla.pkl", "./data/datasets/gowalla/data_complete.npz", False),
}


def _load_location_style_dataset(name, train=True):
    pkl_path, npz_path, labels_one_based = LOCATION_STYLE_DATASETS[name]
    if not os.path.exists(pkl_path):
        dataset = np.load(npz_path)
        x_data = torch.tensor(dataset['x'][:, :]).float()
        y_np = dataset['y'][:]
        if labels_one_based:
            y_np = y_np - 1
        y_data = torch.tensor(y_np).long()
        full_dataset = TensorDataset(x_data, y_data)
        label_counts = np.bincount(y_np.astype(np.int64))
        stratify_labels = y_np if label_counts.size > 0 and label_counts.min() >= 2 else None
        trainset, testset = train_test_split(
            list(range(len(full_dataset))), test_size=0.2,
            random_state=7, stratify=stratify_labels)
        train_dataset = Subset(full_dataset, trainset)
        test_dataset = Subset(full_dataset, testset)
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, 'wb') as f:
            pickle.dump([train_dataset, test_dataset], f)
    else:
        with open(pkl_path, 'rb') as f:
            train_dataset, test_dataset = pickle.load(f)
    return train_dataset if train else test_dataset


def _convert_to_rgb(img):
    """Convert PIL image to RGB (handles grayscale images in datasets like Caltech101)."""
    return img.convert('RGB')


def _fine_grained_train_transform(image_size, mean, std):
    resize_size = image_size + 8
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _imagenet_eval_transform(image_size, mean, std):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _character_train_transform(image_size, mean, std):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _character_eval_transform(image_size, mean, std):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _first_existing_path(paths, description):
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{description} not found. Checked: {paths}")


class STL10Dataset(torch.utils.data.Dataset):
    """
    STL-10 loader that reads the official binary files without torchvision's strict MD5 check.
    Expected local structure:
      root/
        stl10_binary/
          train_X.bin
          train_y.bin
          test_X.bin
          test_y.bin
          class_names.txt
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data_root = os.path.join(root, 'stl10_binary')
        if not os.path.isdir(self.data_root):
            raise RuntimeError(
                "STL-10 not found. Expected folder: "
                f"{self.data_root}. Put stl10_binary under ./data/datasets/stl10-data."
            )

        data_file = 'train_X.bin' if train else 'test_X.bin'
        label_file = 'train_y.bin' if train else 'test_y.bin'
        self.data, self.labels = self._load_file(data_file, label_file)

        class_file = os.path.join(self.data_root, 'class_names.txt')
        if os.path.isfile(class_file):
            with open(class_file, 'r') as f:
                self.classes = [line.strip() for line in f if line.strip()]
        else:
            self.classes = list(range(10))

    def _load_file(self, data_file, label_file):
        data_path = os.path.join(self.data_root, data_file)
        label_path = os.path.join(self.data_root, label_file)
        if not os.path.isfile(data_path) or not os.path.isfile(label_path):
            raise RuntimeError(f"STL-10 split files missing under {self.data_root}.")

        with open(label_path, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8) - 1
        with open(data_path, 'rb') as f:
            images = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(images, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))
        return images, labels

    def __getitem__(self, index):
        image = Image.fromarray(np.transpose(self.data[index], (1, 2, 0)))
        target = int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.data)


class GTSRBDataset(torch.utils.data.Dataset):
    """
    Custom GTSRB loader for the following local structure:
      root/
        GT-final_test.csv                        ← test annotations (semicolon-delimited)
        GTSRB/
          Final_Training/Images/
            00000/  00000_00000.ppm ...  GT-00000.csv
            ...
            00042/  ...
          Final_Test/Images/
            00000.ppm  00001.ppm  ...
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = []  # list of (image_path, class_id)

        if train:
            train_root = os.path.join(root, 'GTSRB', 'Final_Training', 'Images')
            for class_dir in sorted(os.listdir(train_root)):
                class_path = os.path.join(train_root, class_dir)
                if not os.path.isdir(class_path):
                    continue
                class_id = int(class_dir)
                for fname in sorted(os.listdir(class_path)):
                    if fname.lower().endswith('.ppm'):
                        self.samples.append((os.path.join(class_path, fname), class_id))
        else:
            test_images_root = os.path.join(root, 'GTSRB', 'Final_Test', 'Images')
            csv_path = os.path.join(root, 'GT-final_test.csv')
            df = pd.read_csv(csv_path, sep=';')
            for _, row in df.iterrows():
                img_path = os.path.join(test_images_root, row['Filename'])
                self.samples.append((img_path, int(row['ClassId'])))

        self.classes = list(range(43))

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


class Flowers102Dataset(torch.utils.data.Dataset):
    """
    Custom Oxford 102 Flowers loader compatible with torchvision < 0.13.
    Expected local structure:
      root/
        flowers-102/
          jpg/
            image_00001.jpg  ...  image_08189.jpg
          imagelabels.mat   <- 1-based class labels, shape (1, 8189)
          setid.mat         <- trnid / valid / tstid arrays
    Labels are remapped from 1-based to 0-based (0-101).
    split: 'train' uses trnid, 'val' uses valid, 'test' uses tstid,
           'all' combines all three (for pickle-based custom split).
    """
    def __init__(self, root, split='train', transform=None):
        assert _loadmat is not None, "scipy is required for Flowers102Dataset (pip install scipy)"
        self.transform = transform
        base = os.path.join(root, 'flowers-102')
        labels_all = _loadmat(os.path.join(base, 'imagelabels.mat'))['labels'][0]  # shape (8189,)
        setid = _loadmat(os.path.join(base, 'setid.mat'))
        if split == 'train':
            indices = setid['trnid'][0]   # 1-based image indices
        elif split == 'val':
            indices = setid['valid'][0]
        elif split == 'test':
            indices = setid['tstid'][0]
        else:  # 'all': combine trnid + valid + tstid for a custom 80/20 split
            indices = np.concatenate([setid['trnid'][0], setid['valid'][0], setid['tstid'][0]])
        img_dir = os.path.join(base, 'jpg')
        self.samples = []
        for idx in indices:
            fname = f'image_{idx:05d}.jpg'
            label = int(labels_all[idx - 1]) - 1  # 0-based
            self.samples.append((os.path.join(img_dir, fname), label))
        self.classes = list(range(102))

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


class FGVCAircraftDataset(torch.utils.data.Dataset):
    """
    FGVC-Aircraft variant-level classification loader.
    Expected local structure after extracting fgvc-aircraft-2013b.tar.gz:
      root/
        fgvc-aircraft-2013b/
          data/
            images/
              0034309.jpg ...
            variants.txt
            images_variant_train.txt
            images_variant_val.txt
            images_variant_trainval.txt
            images_variant_test.txt
    Labels are mapped to 0-99 using variants.txt.
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = []

        data_root = os.path.join(root, 'fgvc-aircraft-2013b', 'data')
        if not os.path.isdir(data_root):
            raise RuntimeError(
                "FGVC-Aircraft not found. Expected folder: "
                f"{data_root}. Run setup_fgvc_aircraft_dtd.ps1 locally and sync it to the server."
            )

        variants_path = os.path.join(data_root, 'variants.txt')
        with open(variants_path, 'r') as f:
            self.classes = [line.strip() for line in f if line.strip()]
        self.class_to_idx = dict((class_name, idx) for idx, class_name in enumerate(self.classes))

        split_file = 'images_variant_trainval.txt' if train else 'images_variant_test.txt'
        split_path = os.path.join(data_root, split_file)
        if train and not os.path.exists(split_path):
            split_paths = [
                os.path.join(data_root, 'images_variant_train.txt'),
                os.path.join(data_root, 'images_variant_val.txt'),
            ]
        else:
            split_paths = [split_path]

        image_root = os.path.join(data_root, 'images')
        for path in split_paths:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    image_id, label = line.split(maxsplit=1)
                    self.samples.append((
                        os.path.join(image_root, f'{image_id}.jpg'),
                        self.class_to_idx[label],
                    ))

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


class DTDDataset(torch.utils.data.Dataset):
    """
    Describable Textures Dataset loader.
    Expected local structure after extracting dtd-r1.0.1.tar.gz:
      root/
        dtd/
          images/
            banded/...
            blotchy/...
          labels/
            train1.txt
            val1.txt
            test1.txt
    train=True uses train1+val1; train=False uses test1.
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = []

        data_root = os.path.join(root, 'dtd')
        if not os.path.isdir(data_root):
            raise RuntimeError(
                "DTD not found. Expected folder: "
                f"{data_root}. Run setup_fgvc_aircraft_dtd.ps1 locally and sync it to the server."
            )

        image_root = os.path.join(data_root, 'images')
        self.classes = sorted([
            name for name in os.listdir(image_root)
            if os.path.isdir(os.path.join(image_root, name))
        ])
        self.class_to_idx = dict((class_name, idx) for idx, class_name in enumerate(self.classes))

        label_root = os.path.join(data_root, 'labels')
        split_files = ['train1.txt', 'val1.txt'] if train else ['test1.txt']
        for split_file in split_files:
            with open(os.path.join(label_root, split_file), 'r') as f:
                for line in f:
                    rel_path = line.strip()
                    if not rel_path:
                        continue
                    class_name = rel_path.split('/')[0]
                    self.samples.append((
                        os.path.join(image_root, rel_path.replace('/', os.sep)),
                        self.class_to_idx[class_name],
                    ))

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


class OxfordIIITPetDataset(torch.utils.data.Dataset):
    """
    Oxford-IIIT Pet breed classification loader.
    Expected local structure after extracting images.tar.gz and annotations.tar.gz:
      root/
        images/
          Abyssinian_1.jpg ...
        annotations/
          trainval.txt
          test.txt
    Labels are remapped from 1-based to 0-based (0-36).
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = []
        self.classes = list(range(37))

        image_root = os.path.join(root, 'images')
        annotation_root = os.path.join(root, 'annotations')
        if not os.path.isdir(image_root) or not os.path.isdir(annotation_root):
            raise RuntimeError(
                "Oxford-IIIT Pet not found. Expected folders: "
                f"{image_root} and {annotation_root}. Run setup_pet_dogs.ps1 locally and sync it to the server."
            )

        split_file = 'trainval.txt' if train else 'test.txt'
        split_path = os.path.join(annotation_root, split_file)
        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                image_name = parts[0]
                label = int(parts[1]) - 1
                self.samples.append((os.path.join(image_root, f'{image_name}.jpg'), label))

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


def _matlab_string(value):
    while isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.item()
        else:
            value = value[0]
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return str(value)


class StanfordDogsDataset(torch.utils.data.Dataset):
    """
    Stanford Dogs breed classification loader.
    Expected local structure after extracting images.tar and lists.tar:
      root/
        Images/
          n02085620-Chihuahua/...
          ...
        train_list.mat
        test_list.mat
    Labels are remapped from 1-based to 0-based (0-119).
    """
    def __init__(self, root, train=True, transform=None):
        assert _loadmat is not None, "scipy is required for StanfordDogsDataset (pip install scipy)"
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = []

        image_root = os.path.join(root, 'Images')
        split_path = os.path.join(root, 'train_list.mat' if train else 'test_list.mat')
        if not os.path.isdir(image_root) or not os.path.exists(split_path):
            raise RuntimeError(
                "Stanford Dogs not found. Expected folder/file: "
                f"{image_root} and {split_path}. Run setup_pet_dogs.ps1 locally and sync it to the server."
            )

        self.classes = sorted([
            name for name in os.listdir(image_root)
            if os.path.isdir(os.path.join(image_root, name))
        ])
        split = _loadmat(split_path)
        file_list = split['file_list'].reshape(-1)
        labels = split['labels'].reshape(-1)
        for rel_path, label in zip(file_list, labels):
            rel_path = _matlab_string(rel_path).replace('/', os.sep)
            self.samples.append((os.path.join(image_root, rel_path), int(label) - 1))

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


class Food101Dataset(torch.utils.data.Dataset):
    """
    Food-101 loader using the official train/test split.
    Expected local structure after extracting food-101.tar.gz:
      root/
        food-101/
          images/
            apple_pie/...
            ...
          meta/
            classes.txt
            train.txt
            test.txt
    train=True uses meta/train.txt; train=False uses meta/test.txt.
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = []

        data_root = os.path.join(root, 'food-101')
        image_root = os.path.join(data_root, 'images')
        meta_root = os.path.join(data_root, 'meta')
        if not os.path.isdir(image_root) or not os.path.isdir(meta_root):
            raise RuntimeError(
                "Food-101 not found. Expected folders: "
                f"{image_root} and {meta_root}. Run setup_food101_resisc45.ps1 locally and sync it to the server."
            )

        classes_path = os.path.join(meta_root, 'classes.txt')
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f if line.strip()]
        else:
            self.classes = sorted([
                name for name in os.listdir(image_root)
                if os.path.isdir(os.path.join(image_root, name))
            ])
        self.class_to_idx = dict((class_name, idx) for idx, class_name in enumerate(self.classes))

        split_file = 'train.txt' if train else 'test.txt'
        split_path = os.path.join(meta_root, split_file)
        with open(split_path, 'r') as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:
                    continue
                class_name = rel_path.split('/')[0]
                self.samples.append((
                    os.path.join(image_root, rel_path.replace('/', os.sep) + '.jpg'),
                    self.class_to_idx[class_name],
                ))

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


class RESISC45Dataset(torch.utils.data.Dataset):
    """
    NWPU-RESISC45 scene classification loader.
    Expected local structure after extracting NWPU-RESISC45.zip:
      root/
        NWPU-RESISC45/
          airplane/airplane_001.jpg ...
          ...
        resisc45-train.txt  (optional)
        resisc45-val.txt    (optional)
        resisc45-test.txt   (optional)
    If split txt files exist, train=True uses train+val and train=False uses test.
    Otherwise, a deterministic 80/20 stratified split is created from class folders.
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = []

        data_root = os.path.join(root, 'NWPU-RESISC45')
        if not os.path.isdir(data_root):
            raise RuntimeError(
                "NWPU-RESISC45 not found. Expected folder: "
                f"{data_root}. Run setup_food101_resisc45.ps1 locally and sync it to the server."
            )

        self.classes = sorted([
            name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name))
        ])
        self.class_to_idx = dict((class_name, idx) for idx, class_name in enumerate(self.classes))

        file_index = {}
        class_files = []
        for class_name in self.classes:
            class_root = os.path.join(data_root, class_name)
            files = [
                os.path.join(class_root, fname)
                for fname in sorted(os.listdir(class_root))
                if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            class_files.append((class_name, files))
            for path in files:
                file_index[os.path.basename(path)] = (path, self.class_to_idx[class_name])

        if self._has_split_files():
            split_files = ['resisc45-train.txt', 'resisc45-val.txt'] if train else ['resisc45-test.txt']
            for split_file in split_files:
                with open(os.path.join(root, split_file), 'r') as f:
                    for line in f:
                        basename = os.path.basename(line.strip())
                        if basename:
                            self.samples.append(file_index[basename])
        else:
            for class_name, files in class_files:
                train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
                selected_files = train_files if train else test_files
                label = self.class_to_idx[class_name]
                self.samples.extend((path, label) for path in selected_files)

    def _has_split_files(self):
        return all(os.path.exists(os.path.join(self.root, f'resisc45-{split}.txt'))
                   for split in ['train', 'val', 'test'])

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


class PatternNetDataset(torch.utils.data.Dataset):
    """
    PatternNet remote-sensing scene classification loader.
    Expected local structure after extracting PatternNet.zip:
      root/
        PatternNet/
          images/
            airplane/airplane001.jpg ...
            ...
    PatternNet has no official split in this project, so a deterministic
    per-class 80/20 train/test split is generated from class folders.
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = []

        data_root = os.path.join(root, 'PatternNet', 'images')
        if not os.path.isdir(data_root):
            raise RuntimeError(
                "PatternNet not found. Expected folder: "
                f"{data_root}. Run setup_food101_patternnet.ps1 locally and sync it to the server."
            )

        self.classes = sorted([
            name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name))
        ])
        self.class_to_idx = dict((class_name, idx) for idx, class_name in enumerate(self.classes))

        for class_name in self.classes:
            class_root = os.path.join(data_root, class_name)
            files = [
                os.path.join(class_root, fname)
                for fname in sorted(os.listdir(class_root))
                if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
            selected_files = train_files if train else test_files
            label = self.class_to_idx[class_name]
            self.samples.extend((path, label) for path in selected_files)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


class PlantVillageDataset(torch.utils.data.Dataset):
    """
    PlantVillage crop leaf disease classification loader.
    Expected local structure after running setup_plantvillage_officehome.ps1:
      root/
        PlantVillage/
          images/
            Apple___Apple_scab/...
            ...
    The original dataset has no single split for this project, so a deterministic
    per-class 80/20 train/test split is generated from class folders.
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = []

        data_root = os.path.join(root, 'PlantVillage', 'images')
        if not os.path.isdir(data_root):
            raise RuntimeError(
                "PlantVillage not found. Expected folder: "
                f"{data_root}. Run setup_plantvillage_officehome.ps1 locally and sync it to the server."
            )

        self.classes = sorted([
            name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name))
        ])
        self.class_to_idx = dict((class_name, idx) for idx, class_name in enumerate(self.classes))

        for class_name in self.classes:
            class_root = os.path.join(data_root, class_name)
            files = [
                os.path.join(class_root, fname)
                for fname in sorted(os.listdir(class_root))
                if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
            selected_files = train_files if train else test_files
            label = self.class_to_idx[class_name]
            self.samples.extend((path, label) for path in selected_files)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


class OfficeHomeDataset(torch.utils.data.Dataset):
    """
    Office-Home object recognition loader.
    Expected local structure after running setup_plantvillage_officehome.ps1:
      root/
        OfficeHome/
          images/
            Alarm_Clock/...
            ...
    Images from the four domains are combined into the same class folders.
    A deterministic per-class 80/20 train/test split is generated locally.
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = []

        data_root = os.path.join(root, 'OfficeHome', 'images')
        if not os.path.isdir(data_root):
            raise RuntimeError(
                "Office-Home not found. Expected folder: "
                f"{data_root}. Run setup_plantvillage_officehome.ps1 locally and sync it to the server."
            )

        self.classes = sorted([
            name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name))
        ])
        self.class_to_idx = dict((class_name, idx) for idx, class_name in enumerate(self.classes))

        for class_name in self.classes:
            class_root = os.path.join(data_root, class_name)
            files = []
            for dirpath, _, filenames in os.walk(class_root):
                for fname in sorted(filenames):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        files.append(os.path.join(dirpath, fname))
            files = sorted(files)
            train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
            selected_files = train_files if train else test_files
            label = self.class_to_idx[class_name]
            self.samples.extend((path, label) for path in selected_files)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


class MITIndoor67Dataset(torch.utils.data.Dataset):
    """
    MIT Indoor 67 scene classification loader.
    Expected local structure after running setup_mit_indoor67_caltech256.ps1:
      root/
        MITIndoor67/
          images/
            airport_inside/...
            ...
    The full image folder is used with a deterministic per-class 80/20 split
    instead of the smaller 80-train/20-test official protocol.
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = []

        data_root = os.path.join(root, 'MITIndoor67', 'images')
        if not os.path.isdir(data_root):
            raise RuntimeError(
                "MIT Indoor 67 not found. Expected folder: "
                f"{data_root}. Run setup_mit_indoor67_caltech256.ps1 locally and sync it to the server."
            )

        self.classes = sorted([
            name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name))
        ])
        self.class_to_idx = dict((class_name, idx) for idx, class_name in enumerate(self.classes))

        for class_name in self.classes:
            class_root = os.path.join(data_root, class_name)
            files = [
                os.path.join(class_root, fname)
                for fname in sorted(os.listdir(class_root))
                if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
            selected_files = train_files if train else test_files
            label = self.class_to_idx[class_name]
            self.samples.extend((path, label) for path in selected_files)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


class Caltech256BalancedDataset(torch.utils.data.Dataset):
    """
    Balanced Caltech-256 subset loader.
    Expected local structure after running setup_mit_indoor67_caltech256.ps1:
      root/
        Caltech256Balanced/
          images/
            ak47/...
            ...
    The deployment script selects a fixed class-balanced subset. A deterministic
    per-class 80/20 train/test split is generated locally.
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = []

        data_root = os.path.join(root, 'Caltech256Balanced', 'images')
        if not os.path.isdir(data_root):
            raise RuntimeError(
                "Caltech-256 balanced subset not found. Expected folder: "
                f"{data_root}. Run setup_mit_indoor67_caltech256.ps1 locally and sync it to the server."
            )

        self.classes = sorted([
            name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name))
        ])
        self.class_to_idx = dict((class_name, idx) for idx, class_name in enumerate(self.classes))

        for class_name in self.classes:
            class_root = os.path.join(data_root, class_name)
            files = [
                os.path.join(class_root, fname)
                for fname in sorted(os.listdir(class_root))
                if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
            selected_files = train_files if train else test_files
            label = self.class_to_idx[class_name]
            self.samples.extend((path, label) for path in selected_files)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


class ClassFolderSplitDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, train=True, transform=None, test_size=0.2, seed=42):
        self.data_root = data_root
        self.train = train
        self.transform = transform
        self.samples = []

        if not os.path.isdir(data_root):
            raise RuntimeError(f"Class-folder dataset not found. Expected folder: {data_root}")

        self.classes = sorted([
            name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name))
        ])
        self.class_to_idx = dict((class_name, idx) for idx, class_name in enumerate(self.classes))

        for class_name in self.classes:
            class_root = os.path.join(data_root, class_name)
            files = []
            for dirpath, _, filenames in os.walk(class_root):
                for fname in sorted(filenames):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        files.append(os.path.join(dirpath, fname))
            files = sorted(files)
            if len(files) < 2:
                continue
            train_files, test_files = train_test_split(files, test_size=test_size, random_state=seed)
            selected_files = train_files if train else test_files
            label = self.class_to_idx[class_name]
            self.samples.extend((path, label) for path in selected_files)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


class MiniImageNetDataset(ClassFolderSplitDataset):
    """
    Mini-ImageNet loader.
    Expected local structure after running setup_miniimagenet_nico.ps1:
      root/
        MiniImageNet/
          images/
            class_000/...
            ...
    The full class-folder dataset is split deterministically per class.
    """
    def __init__(self, root, train=True, transform=None):
        data_root = os.path.join(root, 'MiniImageNet', 'images')
        if not os.path.isdir(data_root):
            raise RuntimeError(
                "Mini-ImageNet not found. Expected folder: "
                f"{data_root}. Run setup_miniimagenet_nico.ps1 locally and sync it to the server."
            )
        super().__init__(data_root=data_root, train=train, transform=transform)


class NICODataset(ClassFolderSplitDataset):
    """
    NICO object-context classification loader.
    Expected local structure after running setup_miniimagenet_nico.ps1:
      root/
        NICO/
          images/
            bear/...
            ...
    Context folders are flattened into class folders by the deployment script.
    """
    def __init__(self, root, train=True, transform=None):
        data_root = os.path.join(root, 'NICO', 'images')
        if not os.path.isdir(data_root):
            raise RuntimeError(
                "NICO not found. Expected folder: "
                f"{data_root}. Run setup_miniimagenet_nico.ps1 locally and sync it to the server."
            )
        super().__init__(data_root=data_root, train=train, transform=transform)


class Caltech10Dataset(ClassFolderSplitDataset):
    """
    Caltech-10 loader.
    Expected local structure after running setup_caltech10.ps1:
      root/
        Caltech10/
          images/
            airplanes/...
            ...
    The deployment script builds this 10-class subset from Caltech-101.
    """
    def __init__(self, root, train=True, transform=None):
        data_root = os.path.join(root, 'Caltech10', 'images')
        if not os.path.isdir(data_root):
            raise RuntimeError(
                "Caltech-10 not found. Expected folder: "
                f"{data_root}. Run setup_caltech10.ps1 locally and sync it to the server."
            )
        super().__init__(data_root=data_root, train=train, transform=transform)


class Office10Dataset(ClassFolderSplitDataset):
    """
    Office-Caltech-10 / Office-10 loader.
    Expected local structure after running setup_office10.ps1:
      root/
        Office10/
          images/
            back_pack/...
            ...
    Images from Amazon, DSLR, Webcam, and Caltech domains are combined into
    the same 10 class folders.
    """
    def __init__(self, root, train=True, transform=None):
        data_root = os.path.join(root, 'Office10', 'images')
        if not os.path.isdir(data_root):
            raise RuntimeError(
                "Office-10 not found. Expected folder: "
                f"{data_root}. Run setup_office10.ps1 locally and sync it to the server."
            )
        super().__init__(data_root=data_root, train=train, transform=transform)


class Kuzushiji49Dataset(torch.utils.data.Dataset):
    """
    Kuzushiji-49 loader.
    Expected local structure after running setup_kuzushiji49_emnist.ps1:
      root/
        Kuzushiji49/
          k49-train-imgs.npz
          k49-train-labels.npz
          k49-test-imgs.npz
          k49-test-labels.npz
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.classes = list(range(49))

        data_root = os.path.join(root, 'Kuzushiji49')
        split = 'train' if train else 'test'
        images_path = os.path.join(data_root, f'k49-{split}-imgs.npz')
        labels_path = os.path.join(data_root, f'k49-{split}-labels.npz')
        if not os.path.isfile(images_path) or not os.path.isfile(labels_path):
            raise RuntimeError(
                "Kuzushiji-49 not found. Expected files under "
                f"{data_root}. Run setup_kuzushiji49_emnist.ps1 locally and sync it to the server."
            )

        self.data = np.load(images_path)['arr_0'].astype(np.uint8)
        self.targets = np.load(labels_path)['arr_0'].astype(np.int64)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index], mode='L').convert('RGB')
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.targets)


class EMNISTBalancedDataset(torch.utils.data.Dataset):
    """
    EMNIST Balanced loader for the official gzip/IDX files.
    Expected local structure after running setup_kuzushiji49_emnist.ps1:
      root/
        EMNISTBalanced/
          gzip/
            emnist-balanced-train-images-idx3-ubyte.gz
            emnist-balanced-train-labels-idx1-ubyte.gz
            emnist-balanced-test-images-idx3-ubyte.gz
            emnist-balanced-test-labels-idx1-ubyte.gz
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.classes = list(range(47))

        data_root = os.path.join(root, 'EMNISTBalanced', 'gzip')
        split = 'train' if train else 'test'
        images_path = os.path.join(data_root, f'emnist-balanced-{split}-images-idx3-ubyte.gz')
        labels_path = os.path.join(data_root, f'emnist-balanced-{split}-labels-idx1-ubyte.gz')
        if not os.path.isfile(images_path) or not os.path.isfile(labels_path):
            raise RuntimeError(
                "EMNIST Balanced not found. Expected gzip IDX files under "
                f"{data_root}. Run setup_kuzushiji49_emnist.ps1 locally and sync it to the server."
            )

        self.data = self._read_idx_images(images_path)
        self.targets = self._read_idx_labels(labels_path)

    @staticmethod
    def _read_idx_images(path):
        with gzip.open(path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise RuntimeError(f"Invalid IDX image file: {path}")
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)

    @staticmethod
    def _read_idx_labels(path):
        with gzip.open(path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise RuntimeError(f"Invalid IDX label file: {path}")
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        if labels.shape[0] != num:
            raise RuntimeError(f"Invalid IDX label length in {path}")
        return labels.astype(np.int64)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index], mode='L').convert('RGB')
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.targets)


class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, size=64):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        folder_name = 'tiny-imagenet-200-32' if int(size) == 32 else 'tiny-imagenet-200'
        dataset_root = os.path.join(root, folder_name)
        if not os.path.isdir(dataset_root):
            raise RuntimeError(
                "TinyImageNet not found. Expected folder: "
                f"{dataset_root}. Download tiny-imagenet-200 and place it there."
            )

        wnids_path = os.path.join(dataset_root, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            self.classes = [line.strip() for line in f if line.strip()]
        self.class_to_idx = dict((cls_name, idx) for idx, cls_name in enumerate(self.classes))

        if train:
            train_root = os.path.join(dataset_root, 'train')
            for cls_name in self.classes:
                image_root = os.path.join(train_root, cls_name, 'images')
                if not os.path.isdir(image_root):
                    continue
                for image_name in sorted(os.listdir(image_root)):
                    if image_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                        self.samples.append((os.path.join(image_root, image_name), self.class_to_idx[cls_name]))
        else:
            val_root = os.path.join(dataset_root, 'val')
            annotation_path = os.path.join(val_root, 'val_annotations.txt')
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    image_name, cls_name = parts[0], parts[1]
                    if cls_name in self.class_to_idx:
                        self.samples.append((os.path.join(val_root, 'images', image_name), self.class_to_idx[cls_name]))

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)



def get_dataset(name, train=True, augment=None):
    name = str(name).strip().lower().replace('-', '_')
    if name == "texas":
        name = "texas100"
    elif name == "purchase":
        name = "purchase100"
    elif name == "foursquare":
        name = "foursquare_nyc"
    if augment is None:
        augment = train
    print(f"Build Dataset {name} (train={train}, augment={augment})")
    if name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data/datasets/cifar10-data', train=train, download=True, transform=transform)
    elif name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR100(root='./data/datasets/cifar100-data', train=train, download=True, transform=transform)
    elif name == "tinyimagenet":
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        tiny_size = int(os.environ.get('TINYIMAGENET_SIZE', '64'))
        if train and augment:
            transform = _fine_grained_train_transform(tiny_size, mean, std)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        dataset = TinyImageNetDataset(root='./data/datasets/tinyimagenet', train=train, transform=transform, size=tiny_size)
    elif name == "mnist":
        mean = (0.1307,)
        std = (0.3081,)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
                                        ])
        dataset = torchvision.datasets.MNIST(root='data/datasets/mnist-data', train=train, download=True,
                                             transform=transform)
    elif name == "cinic":
        # the dataset can be downloaded from https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz?sequence=4&isAllowed=y
        if not os.path.exists("./data/datasets/cinic/cinic.pkl"):
            cinic_directory = './data/datasets/cinic'
            cinic_mean = [0.47889522, 0.47227842, 0.43047404]
            cinic_std = [0.24205776, 0.23828046, 0.25874835]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=cinic_mean,std=cinic_std),
            ])
            trainset = torchvision.datasets.ImageFolder(cinic_directory + '/train', transform=transform)
            testset = torchvision.datasets.ImageFolder(cinic_directory + '/test', transform=transform)

            with open("./data/datasets/cinic/train_test_idx.pkl", 'rb') as f:
                trainidx, testidx= pickle.load(f)
            train_data = Subset(trainset, trainidx)
            test_data = Subset(testset, testidx)
            with open("./data/datasets/cinic/cinic.pkl", "wb") as f:
                pickle.dump([train_data, test_data], f)
        else:
            with open("./data/datasets/cinic/cinic.pkl", "rb") as f:
                train_data, test_data = pickle.load(f)
        if train == False:
            dataset = test_data
        else:
            dataset = train_data
    
    elif name == "texas100":
        # the dataset can be downloaded from https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz
        if not os.path.exists("./data/datasets/texas/texas100.pkl"):
            feats_path = _first_existing_path([
                "./data/datasets/texas/feats.txt",
                "./data/datasets/texas/texas/100/feats",
            ], "Texas100 features")
            labels_path = _first_existing_path([
                "./data/datasets/texas/labels.txt",
                "./data/datasets/texas/texas/100/labels",
            ], "Texas100 labels")
            x = np.loadtxt(feats_path, delimiter=',')
            x_data = torch.tensor(x[:, :]).float()
            y = np.loadtxt(labels_path, delimiter=',')
            y_data = torch.tensor(y[:] - 1).long()
            dataset = TensorDataset(x_data, y_data)
            trainset, testset = train_test_split(list(range(len(dataset))), test_size=0.2) # Make sure to calculate the mem-score for these training data.
            train_dataset = Subset(dataset, trainset)
            test_dataset = Subset(dataset, testset)
            with open("./data/datasets/texas/texas100.pkl", 'wb') as f:
                pickle.dump([train_dataset, test_dataset], f)
        else:
            with open("./data/datasets/texas/texas100.pkl", 'rb') as f:
                train_dataset, test_dataset = pickle.load(f)
        if train == False:
            dataset = test_dataset
        else:
            dataset = train_dataset

    elif name in LOCATION_STYLE_DATASETS:
        # Location-style user check-in datasets use x/y arrays stored in data_complete.npz.
        dataset = _load_location_style_dataset(name, train=train)

    elif name == "purchase100":
        # the dataset can be downloaded from https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz
        if not os.path.exists("./data/datasets/purchase100/purchase100.pkl"):
            purchase_path = _first_existing_path([
                "./data/datasets/purchase100/purchase100.txt",
                "./data/datasets/purchase100/dataset_purchase.txt",
            ], "Purchase100 data")
            dataset = np.loadtxt(purchase_path, delimiter=',')
            x_data = torch.tensor(dataset[:, 1:], dtype=torch.float32)
            y_data = torch.tensor(dataset[:, 0] - 1, dtype=torch.long)
            dataset = TensorDataset(x_data, y_data)
            trainset, testset = train_test_split(list(range(len(dataset))), test_size=0.2) # Make sure to calculate the mem-score for these training data.
            train_dataset = Subset(dataset, trainset)
            test_dataset = Subset(dataset, testset)
            with open("./data/datasets/purchase100/purchase100.pkl", 'wb') as f:
                pickle.dump([train_dataset, test_dataset], f)
        else:
            with open("./data/datasets/purchase100/purchase100.pkl", 'rb') as f:
                train_dataset, test_dataset = pickle.load(f)
        if train == False:
            dataset = test_dataset
        else:
            dataset = train_dataset

    elif name == "gtsrb":
        # German Traffic Sign Recognition Benchmark: 43 classes, images resized to 32x32
        mean = (0.3403, 0.3121, 0.3214)
        std = (0.2724, 0.2608, 0.2669)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = GTSRBDataset(root='./data/datasets/gtsrb', train=train, transform=transform)

    elif name == "flowers102":
        # Oxford 102 Flower Categories: 102 classes, images resized to 32x32
        # Uses all 8189 images with a pickle-cached 80/20 split (standard trnid has only
        # 10 imgs/class which is too few for the sliding-window defence mechanism).
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        pkl_path = './data/datasets/flowers102_split.pkl'
        if not os.path.exists(pkl_path):
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            full_dataset = Flowers102Dataset(root='./data/datasets', split='all', transform=transform)
            trainset_idx, testset_idx = train_test_split(
                list(range(len(full_dataset))), test_size=0.2, random_state=42)
            train_dataset = Subset(full_dataset, trainset_idx)
            test_dataset = Subset(full_dataset, testset_idx)
            with open(pkl_path, 'wb') as f:
                pickle.dump([train_dataset, test_dataset], f)
        else:
            with open(pkl_path, 'rb') as f:
                train_dataset, test_dataset = pickle.load(f)
        dataset = train_dataset if train else test_dataset

    elif name == "caltech101":
        # Caltech-101: 101 classes, no standard split — use pickle-cached train/test split
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        pkl_path = "./data/datasets/caltech101/caltech101.pkl"
        if not os.path.exists(pkl_path):
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Lambda(_convert_to_rgb),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            full_dataset = torchvision.datasets.Caltech101(
                root='./data/datasets/caltech101', download=True, transform=transform)
            trainset_idx, testset_idx = train_test_split(
                list(range(len(full_dataset))), test_size=0.2, random_state=42)
            train_dataset = Subset(full_dataset, trainset_idx)
            test_dataset = Subset(full_dataset, testset_idx)
            os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            with open(pkl_path, 'wb') as f:
                pickle.dump([train_dataset, test_dataset], f)
        else:
            with open(pkl_path, 'rb') as f:
                train_dataset, test_dataset = pickle.load(f)
        dataset = train_dataset if train else test_dataset

    elif name == "caltech10":
        # Caltech-10: fixed 10-class Caltech-101 subset, resized to 96x96.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if train and augment:
            transform = _fine_grained_train_transform(CALTECH10_IMAGE_SIZE, mean, std)
        else:
            transform = transforms.Compose([
                transforms.Resize((CALTECH10_IMAGE_SIZE, CALTECH10_IMAGE_SIZE)),
                transforms.Lambda(_convert_to_rgb),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        dataset = Caltech10Dataset(root='./data/datasets/caltech10', train=train, transform=transform)

    elif name == "fgvc_aircraft":
        # FGVC-Aircraft: 100 fine-grained aircraft variants, resized to 64x64.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if train and augment:
            transform = _fine_grained_train_transform(FGVC_AIRCRAFT_IMAGE_SIZE, mean, std)
        else:
            transform = _imagenet_eval_transform(FGVC_AIRCRAFT_IMAGE_SIZE, mean, std)
        dataset = FGVCAircraftDataset(
            root='./data/datasets/fgvc_aircraft', train=train, transform=transform)

    elif name == "dtd":
        # Describable Textures Dataset: 47 texture classes, resized to 64x64.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((DTD_IMAGE_SIZE, DTD_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = DTDDataset(root='./data/datasets/dtd', train=train, transform=transform)

    elif name == "oxford_pet":
        # Oxford-IIIT Pet: 37 pet breeds, standard trainval/test split, resized to 64x64.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if train and augment:
            transform = _fine_grained_train_transform(OXFORD_PET_IMAGE_SIZE, mean, std)
        else:
            transform = _imagenet_eval_transform(OXFORD_PET_IMAGE_SIZE, mean, std)
        dataset = OxfordIIITPetDataset(root='./data/datasets/oxford_pet', train=train, transform=transform)

    elif name == "stanford_dogs":
        # Stanford Dogs: 120 dog breeds, standard train/test split, resized to 64x64.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if train and augment:
            transform = _fine_grained_train_transform(STANFORD_DOGS_IMAGE_SIZE, mean, std)
        else:
            transform = _imagenet_eval_transform(STANFORD_DOGS_IMAGE_SIZE, mean, std)
        dataset = StanfordDogsDataset(root='./data/datasets/stanford_dogs', train=train, transform=transform)

    elif name == "food101":
        # Food-101: 101 classes, 750 train + 250 test images per class, resized to 64x64.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((FOOD101_IMAGE_SIZE, FOOD101_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Food101Dataset(root='./data/datasets/food101', train=train, transform=transform)

    elif name == "resisc45":
        # NWPU-RESISC45: 45 remote-sensing scene classes, 700 images per class.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((RESISC45_IMAGE_SIZE, RESISC45_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = RESISC45Dataset(root='./data/datasets/resisc45', train=train, transform=transform)

    elif name == "patternnet":
        # PatternNet: 38 remote-sensing scene classes, 800 images per class.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((PATTERNNET_IMAGE_SIZE, PATTERNNET_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = PatternNetDataset(root='./data/datasets/patternnet', train=train, transform=transform)

    elif name == "plantvillage":
        # PlantVillage: 38 plant species/disease classes, resized to 64x64.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((PLANTVILLAGE_IMAGE_SIZE, PLANTVILLAGE_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = PlantVillageDataset(root='./data/datasets/plantvillage', train=train, transform=transform)

    elif name == "office_home":
        # Office-Home: 65 object classes across four visual domains, resized to 64x64.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((OFFICE_HOME_IMAGE_SIZE, OFFICE_HOME_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = OfficeHomeDataset(root='./data/datasets/office_home', train=train, transform=transform)

    elif name == "office10":
        # Office-Caltech-10: 10 office object classes across four visual domains, resized to 64x64.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if train and augment:
            transform = _fine_grained_train_transform(OFFICE10_IMAGE_SIZE, mean, std)
        else:
            transform = transforms.Compose([
                transforms.Resize((OFFICE10_IMAGE_SIZE, OFFICE10_IMAGE_SIZE)),
                transforms.Lambda(_convert_to_rgb),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
        ])
        dataset = Office10Dataset(root='./data/datasets/office10', train=train, transform=transform)

    elif name == "kuzushiji49":
        # Kuzushiji-49: 49 Japanese cursive character classes, native 28x28 grayscale.
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        if train and augment:
            transform = _character_train_transform(KUZUSHIJI49_IMAGE_SIZE, mean, std)
        else:
            transform = _character_eval_transform(KUZUSHIJI49_IMAGE_SIZE, mean, std)
        dataset = Kuzushiji49Dataset(root='./data/datasets/kuzushiji49', train=train, transform=transform)

    elif name == "emnist_balanced":
        # EMNIST Balanced: 47 balanced handwritten character classes, native 28x28 grayscale.
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        if train and augment:
            transform = _character_train_transform(EMNIST_BALANCED_IMAGE_SIZE, mean, std)
        else:
            transform = _character_eval_transform(EMNIST_BALANCED_IMAGE_SIZE, mean, std)
        dataset = EMNISTBalancedDataset(root='./data/datasets/emnist_balanced', train=train, transform=transform)

    elif name == "mit_indoor67":
        # MIT Indoor 67: 67 indoor scene classes, resized to 96x96.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if train and augment:
            transform = _fine_grained_train_transform(MIT_INDOOR67_IMAGE_SIZE, mean, std)
        else:
            transform = _imagenet_eval_transform(MIT_INDOOR67_IMAGE_SIZE, mean, std)
        dataset = MITIndoor67Dataset(root='./data/datasets/mit_indoor67', train=train, transform=transform)

    elif name == "miniimagenet":
        # Mini-ImageNet: 100 ImageNet-like classes, resized to 84x84.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if train and augment:
            transform = _fine_grained_train_transform(MINIIMAGENET_IMAGE_SIZE, mean, std)
        else:
            transform = _imagenet_eval_transform(MINIIMAGENET_IMAGE_SIZE, mean, std)
        dataset = MiniImageNetDataset(root='./data/datasets/miniimagenet', train=train, transform=transform)

    elif name == "nico":
        # NICO: 19 object classes with diverse contexts, resized to 64x64.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if train and augment:
            transform = _fine_grained_train_transform(NICO_IMAGE_SIZE, mean, std)
        else:
            transform = _imagenet_eval_transform(NICO_IMAGE_SIZE, mean, std)
        dataset = NICODataset(root='./data/datasets/nico', train=train, transform=transform)

    elif name == "caltech256_balanced":
        # Caltech-256 balanced subset: 100 object classes, resized to 96x96.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((CALTECH256_BALANCED_IMAGE_SIZE, CALTECH256_BALANCED_IMAGE_SIZE)),
            transforms.Lambda(_convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = Caltech256BalancedDataset(root='./data/datasets/caltech256_balanced', train=train, transform=transform)

    elif name == "stl10":
        # STL-10: 10 object classes, 500 official train images/class, native 96x96.
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
        dataset = STL10Dataset(root='./data/datasets/stl10-data', train=train, transform=transform)

    else:
        raise ValueError(f"Unsupported dataset name: {name!r}")

    return dataset
