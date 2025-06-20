from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocabulary import Vocabulary

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def train_step(model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    model.train()
    train_loss = 0

    for imgs, caps in data_loader:
        imgs, caps = imgs.to(device), caps.to(device)

        optimizer.zero_grad()

        # Remove <EOS> token from captions while training
        y_out = model(imgs, caps[:, :-1])

        # Remove <SOS> token from captions for calculating loss
        y_true = caps[:, 1:]

        # Calculate loss
        loss = loss_fn(y_out.reshape(-1, y_out.size(2)), y_true.reshape(-1))
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    return train_loss


def validation_step(model: nn.Module,
                    data_loader: torch.utils.data.DataLoader,
                    loss_fn: nn.Module,
                    device: torch.device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for imgs, caps in data_loader:
            imgs, caps = imgs.to(device), caps.to(device)

            # Remove <EOS> token when passing through model
            y_out = model(imgs, caps[:, :-1])

            # Remove <SOS> token from captions for calculating loss
            y_true = caps[:, 1:]

            # Calculate validation loss
            loss = loss_fn(y_out.reshape(-1, y_out.size(2)), y_true.reshape(-1))
            val_loss += loss.item()

    val_loss /= len(data_loader)
    return val_loss


def train_model(model: nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epochs: int,
                checkpoint=False):
    val_loss = []
    train_loss = []
    bleu1_scores = []
    bleu4_scores = []

    for epoch in tqdm(range(epochs)):
        training_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
        validation_loss = validation_step(model, test_dataloader, loss_fn, device)
        train_loss.append(training_loss)
        val_loss.append(validation_loss)

        print(f"Epoch {epoch + 1} | Train loss: {training_loss:.4f} | Val loss: {validation_loss:.4f}")

        if len(val_loss) > 1 and abs(val_loss[-1] - val_loss[-2]) < 5e-6:
            print(f"Validation loss delta below threshold. Stopping training at {epoch + 1} epochs.")
            break

    return dict(train_loss=train_loss, val_loss=val_loss, bleu1=bleu1_scores, bleu4=bleu4_scores)


def make_captions(model: nn.Module,
                  image_tensor: torch.Tensor,
                  vocabulary: Vocabulary,
                  device: torch.device,
                  max_length=50):
    model = model.module if isinstance(model, nn.DataParallel) else model

    model.eval()
    model = model.to(device)

    image_tensor = image_tensor.to(device)
    features = model.encoder(image_tensor)

    caption = [vocabulary.stoi["<SOS>"]]
    for _ in range(max_length):
        caption_tensor = torch.tensor(caption).unsqueeze(0).to(device)
        out = model.decoder(features, caption_tensor)

        top_k = 5
        probs = F.softmax(out[:, -1, :], dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
        chosen = top_k_indices[0][torch.multinomial(top_k_probs[0], 1)].item()
        caption.append(chosen)

        if chosen == vocabulary.stoi["<EOS>"]:
            break

    sentence = " ".join([vocabulary.itos[idx] for idx in caption[1:-1]])
    return sentence

def calculate_bleu(references, candidates):
    n = len(candidates)
    bleu1 = bleu4 = 0.0
    smoothing = SmoothingFunction().method4

    bleu1_weights = (1.0, 0, 0, 0)
    bleu4_weights = (0.25, 0.25, 0.25, 0.25)

    for ref, cand in zip(references, candidates):
        bleu1 += sentence_bleu(ref, cand, weights=bleu1_weights, smoothing_function=smoothing)
        bleu4 += sentence_bleu(ref, cand, weights=bleu4_weights, smoothing_function=smoothing)

    return bleu1 / n, bleu4 / n

def make_captions_beam_search(model: nn.Module,
                              image_tensor: torch.Tensor,
                              vocabulary,
                              device: torch.device,
                              max_length = 50,
                              beam_width = 3):
    if isinstance(model, nn.DataParallel):
        model = model.module

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        features = model.encoder(image_tensor)

    sequences = [([vocabulary.stoi["<SOS>"]], 0.0)]
    skip_tokens = [vocabulary.stoi["<SOS>"], vocabulary.stoi["<EOS>"], vocabulary.stoi["<PAD>"]]

    with torch.no_grad():
        for _ in range(max_length):
            candidates = []

            for seq, score in sequences:
                if seq[-1] == vocabulary.stoi["<EOS>"]:
                    candidates.append((seq, score))
                    continue

                input_tensor = torch.tensor(seq).unsqueeze(0).to(device)
                output = model.decoder(features, input_tensor)
                log_probs = F.softmax(output[:, -1, :], dim=1)
                top_k_log_probs, top_k_idx = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    next_seq = seq + [top_k_idx[0][i].item()]
                    next_score = score + top_k_log_probs[0][i].item()
                    candidates.append((next_seq, next_score))

            # Keeping best k sequences
            sequences = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # Check if all sequences have ended
            if all(seq[0][-1] == vocabulary.stoi["<EOS>"] for seq in sequences):
                break

    # Return best sequence
    best_seq = sequences[0][0]
    return " ".join([vocabulary.itos[token] for token in best_seq if token not in skip_tokens])
