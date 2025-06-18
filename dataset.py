import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from vocabulary import Vocabulary


class Image_Captioning_Dataset(Dataset):
    def __init__(self,
                 root_folder: str,
                 captions_path: str,
                 vocabulary: Vocabulary,
                 train_images=None,
                 transforms=None):
        self.imgs = []
        self.captions = []
        self.tfms = transforms
        self.root_dir = root_folder
        self.vocabulary = vocabulary

        with open(captions_path, "r") as f:
            for line in f:
                if line.startswith("image"):
                    continue

                line = line.strip()

                if line == "":
                    continue

                img, caption = line.split(",", 1)

                if train_images and img not in train_images:
                    continue

                self.imgs.append(img)
                self.captions.append(caption)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.root_dir, "Images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        if self.tfms:
            img = self.tfms(img)

        caption = self.captions[idx]
        tokens = self.vocabulary.numericalize(caption)
        return img, torch.tensor(tokens)

    def collate(self, batch):
        images = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first=True,
                                padding_value=self.vocabulary.stoi["<PAD>"])
        return images, captions