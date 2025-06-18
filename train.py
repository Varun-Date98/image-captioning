import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from vocabulary import Vocabulary
from dataset import Image_Captioning_Dataset
from models.model import ImageCaptioningModel
from utils import train_model, make_captions, calculate_bleu


DATA_DIR = "flickr8k"
IMAGES_DIR = DATA_DIR + "/Images"
CAPTIONS_FILE = DATA_DIR + "/captions.txt"

with open(CAPTIONS_FILE, "r") as f:
    lines = f.readlines()

image_names = [line.split(",")[0] for line in lines]
random.shuffle(image_names)
image_names = list(set(image_names))

n = len(image_names)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

train_images = image_names[:n_train]
val_images = image_names[n_train:n_train+n_val]
test_images = image_names[n_train+n_val:]

print(f"Number of training images: {len(train_images)}")
print(f"Number of validation images: {len(val_images)}")
print(f"Number of testing images: {len(test_images)}")

# Building vocabulary on just the training data
vocab = Vocabulary(freq_threshold=1)
vocab.build_vocabulary_from_captions(CAPTIONS_FILE, training_images=train_images)
print(f"Vocabulary size: {len(vocab)}")

# Defining image transforms
transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Setting up datasets and data loaders
train_dataset = Image_Captioning_Dataset(root_folder=DATA_DIR,
                                         captions_path=CAPTIONS_FILE,
                                         vocabulary=vocab,
                                         train_images=train_images,
                                         transforms=transforms)

val_dataset = Image_Captioning_Dataset(root_folder=DATA_DIR,
                                       captions_path=CAPTIONS_FILE,
                                       vocabulary=vocab,
                                       train_images=val_images,
                                       transforms=transforms)

train_dataloader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True,
                              collate_fn=train_dataset.collate)

val_dataloader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True,
                            collate_fn=train_dataset.collate)

# GPU training
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model invocation
model = ImageCaptioningModel(embed_dim=512,
                             hidden_dim=512,
                             vocab_size=len(vocab),
                             attention_dim=256,
                             train_cnn=False).to(device)

# Setting up optimizer and loss function
pad_idx = vocab.stoi["<PAD>"]
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

# Using data parallel when available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)


results = train_model(model,
                      train_dataloader,
                      val_dataloader,
                      loss_fn,
                      optimizer,
                      device,
                      epochs=10)

# Computing BLEU1 and BLEU4 scores
model.eval()
references = []
candidates = []
skip_tokens = [vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]]

with torch.no_grad():
    for imgs, captions in val_dataloader:
        imgs, captions = imgs.to(device), captions.to(device)

        for i in range(imgs.size(0)):
            img = imgs[i].unsqueeze(0)
            pred_tokens = make_captions(model, img, vocab, device).split()

            original_tokens = captions[i].tolist()
            original_tokens = [vocab.itos[token] for token in original_tokens if token not in skip_tokens]

            references.append([original_tokens])
            candidates.append(pred_tokens)

bleu1, bleu4 = calculate_bleu(references, candidates)
print(f"BLEU1 score: {bleu1:.4f} | BLEU4 score: {bleu4:.4f}")