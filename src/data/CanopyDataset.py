import os
import json
import numpy as np
import pandas as pd
import torch
from skimage.io import imread
from torch.utils.data import Dataset

class CanopyDataset(Dataset):
    def __init__(self, dataset_root='../data/dataset/', split_file='../data/dataset/data_split.csv', split='train', stats_file=None, normalize_labels=True):
        self.normalize_labels = normalize_labels
        self.dataset_root = dataset_root
        self.stats_file = stats_file

        split_df = pd.read_csv(split_file)
        label_folder = 'labels/'
        split_df = (split_df
                        .drop(index=0)  # Image 0 doesn't exist
                        .loc[split_df['split'] == split, :])  # Only the split wanted
        split_df['label_path'] = split_df['label_path'].map(lambda x: label_folder + x.split('/')[1])

        self.data = [(dataset_root + row['image_path'], dataset_root + row['label_path']) for _, row in split_df.iterrows()]

        self.load_or_compute_statistics()

    def __len__(self):
        return len(self.data)

    def load_or_compute_statistics(self):
        if self.stats_file and os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
                self.band_means = np.array(stats['band_means'])
                self.band_stds = np.array(stats['band_stds'])
                self.label_mean = stats['label_mean']
                self.label_std = stats['label_std']
        else:
            self.band_means, self.band_stds = self.compute_band_statistics()
            self.label_mean, self.label_std = self.compute_label_statistics()

    def save_statistics(self, stats_file):
        stats = {
            'band_means': [float(v) for v in self.band_means.tolist()],
            'band_stds': [float(v) for v in self.band_stds.tolist()],
            'label_mean': float(self.label_mean),
            'label_std': float(self.label_std)
        }
        with open(stats_file, 'w') as f:
            json.dump(stats, f)


    def compute_band_statistics(self):
        all_images = np.array([imread(img_path) for img_path, _ in self.data])  # Shape (N, H, W, 12)
        band_means = np.mean(all_images, axis=(0, 1, 2))
        band_stds = np.std(all_images, axis=(0, 1, 2))
        return band_means, band_stds

    def compute_label_statistics(self):
        label_values = [imread(label_path).astype(np.float32) for _, label_path in self.data]
        valid_labels = np.concatenate([label[label != 255] for label in label_values])
        label_mean = np.mean(valid_labels)
        label_std = np.std(valid_labels)
        return label_mean, label_std

    def normalize_image(self, img):
        return (img - self.band_means) / self.band_stds

    def normalize_label(self, label):
        mask = (label != 255).astype(np.float32)  # 1 where valid, 0 where missing
        label = np.where(mask, (label - self.label_mean) / self.label_std, 0)  # Normalize only valid pixels
        return label, mask

    def __getitem__(self, idx):
        img_path, label_path = self.data[idx]
        img = imread(img_path).astype(np.float32)  # Shape (H, W, 12)
        label = imread(label_path).astype(np.float32)  # Shape (H, W)

        img = self.normalize_image(img)
        label, mask = self.normalize_label(label)

        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return img.permute(2, 0, 1), label, mask
