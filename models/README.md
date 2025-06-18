# Image Captioning with CNN Encoder and Attention-based LSTM Decoder

This project implements an end-to-end deep learning pipeline for generating natural language captions for images using a **ResNet-50 encoder** and a **two-layer LSTM decoder with additive attention**. The model is trained on the **Flickr8k** dataset and evaluated using BLEU scores.

---

## Features

- Pretrained ResNet-50 as image encoder
- Additive attention mechanism to focus on relevant image regions
- LSTM-based decoder with skip connections and normalization
- DataLoader with custom vocabulary and padding
- BLEU-1 and BLEU-4 score evaluation

---

## Project Structure

```
.
├── train.py                 # Training and evaluation script
├── dataset.py               # Custom PyTorch Dataset
├── models/
│   ├── encoder.py           # CNN Encoder using ResNet-50
│   ├── decoder.py           # LSTM Decoder with attention
│   ├── model.py             # Combined encoder-decoder model
│   └── attention.py         # Additive attention module
├── vocabulary.py            # Vocabulary builder and tokenizer
├── utils.py                 # Training loop and evaluation utilities
```

---

## Model Architecture

- **Encoder**: ResNet-50 pretrained, returns feature maps of shape `(batch_size, 49, 2048)`
- **Attention Module**: Additive attention (Bahdanau) to compute context vectors
- **Decoder**:
  - Word embedding layer
  - Two-layer LSTM for sequence modeling
  - Attention-augmented context fusion and skip connections
  - LayerNorm and dropout regularization

---

## Dataset

- **Flickr8k** dataset
- 8000 images, each with 5 captions
- Format of `captions.txt`:  
  ```
  image_name.jpg,A caption for the image
  ```

---

## How to Run

### 1. Install Dependencies

```bash
pip install torch torchvision nltk tqdm
```

And download the NLTK tokenizer:

```python
import nltk
nltk.download('punkt')
```

### 2. Prepare Dataset

Create the following folder structure:

```
flickr8k/
├── Images/
│   ├── image1.jpg
│   └── ...
└── captions.txt
```

### 3. Train the Model

```bash
python train.py
```

All key parameters can be edited in `train.py`.

---

## Evaluation

After training, BLEU-1 and BLEU-4 scores are printed for validation data.

Example:
```
BLEU1 score: 0.2300 | BLEU4 score: 0.0480
```

---

## Caption Generation

Captions are generated using greedy decoding with top-k sampling:

```python
make_captions(model, image_tensor, vocabulary, device)
```

---

## Future Improvements

- Use beam search decoding
- Incorporate scheduled sampling
- Try pretrained GloVe embeddings
- Extend to MS COCO dataset

---

## Author

**Varun Date**  
Pull requests, stars, and issues are welcome!

---

## License

This project is licensed under the MIT License.
