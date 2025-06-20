import torch
from PIL import Image
from torchvision import transforms as T
from huggingface_hub import hf_hub_download

import streamlit as st

from vocabulary import Vocabulary
from utils import make_captions_beam_search
from models.model import ImageCaptioningModel


transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def download_model_artifacts():
    repo_id = st.secrets["huggingface"]["repo_id"]
    file_name = st.secrets["huggingface"]["file_name"]

    model_path = hf_hub_download(repo_id, file_name)
    model_artifacts = torch.load(model_path, map_location="cpu", weights_only=False)
    return model_artifacts

def main():
    st.set_page_config(page_title="Image Captioning", layout="centered")

    # 🏷️ Title
    st.title("Caption your Images using Deep Learning Model")

    # Model Description
    st.markdown("""
    This app uses a **pre-trained ResNet-50** encoder and an **additive-attention-based LSTM decoder**  
    to generate captions for images. The model was trained on the **Flickr8k** dataset and achieved a  
    **BLEU-1 score of 0.24**.
    """)

    model_artifacts = download_model_artifacts()
    vocab = model_artifacts["vocabulary"]
    model = ImageCaptioningModel(embed_dim=512,
                                 hidden_dim=512,
                                 vocab_size=len(vocab),
                                 attention_dim=256,
                                 train_cnn=False)
    model.load_state_dict(model_artifacts["model_state_dict"])

    # Image Upload
    uploaded_file = st.file_uploader("Upload an image to caption", type=["jpg", "jpeg", "png"])

    # Display Image and Generate Caption
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        image_tensor = transforms(image).unsqueeze(0)

        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):

                caption = make_captions_beam_search(model, image_tensor, vocab, device=torch.device("cpu"))
                st.success(f"📌 Caption: {caption}")


if __name__ == "__main__":
    main()
