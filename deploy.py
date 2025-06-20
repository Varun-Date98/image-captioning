import streamlit as st
from huggingface_hub import HfApi


repo_id = st.secrets["huggingface"]["repo_id"]
file_name = st.secrets["huggingface"]["file_name"]
model_path = "artifacts/image_captioning_model_lstm.pth"

gateway = HfApi()
gateway.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=file_name,
    repo_id=repo_id,
    repo_type="model"
)
