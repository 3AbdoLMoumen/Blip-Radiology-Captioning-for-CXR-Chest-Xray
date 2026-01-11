from transformers import AutoProcessor, BlipForConditionalGeneration
from utils import *
import numpy as np
from PIL import Image
import streamlit as st
import io
import torch

st.set_page_config(page_title='CHESTXRAY.AI', page_icon = "edma.webp",)

@st.cache_resource
def load_processor_model():
    processor = AutoProcessor.from_pretrained("./processor")
    model = BlipForConditionalGeneration.from_pretrained(
        "./model", 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False
    )
    return processor, model

processor, model = load_processor_model()
st.image("LOGO1.png")

uploaded_file = st.file_uploader("Upload a Chest X-Ray", type=["jpg", "png", "jpeg", "webp"])

sidebar = st.sidebar
sidebar.image("LOGO2.png")
sidebar.text("Disclaimer: Not a medical tool")
max_length = sidebar.slider("Max Tokens", 16, 120, 32, 1)
greedy_mode = sidebar.checkbox("Greedy Mode")

if greedy_mode:
    temperature = 0 
    top_p = 1
    do_sample=False
else:
	do_sample=True
	temperature = sidebar.slider("Temperature", 0.1, 1., 0.3, 0.01)
	top_p = sidebar.slider("Top P", 0., 1., 0.9, 0.01)

if uploaded_file is not None:
	file_bytes = io.BytesIO(uploaded_file.read())
	image = Image.open(file_bytes)
	image.thumbnail((128, 128))  
	left_co, cent_co,last_co = st.columns(3)
	with cent_co:
		st.image(image, caption="Uploaded Image", width=212)
		left_co_, cent_co_,last_co_ = cent_co.columns(3)
		with cent_co_:
			button = st.button("Run")
	if button:
		prediction = predict(image, processor, model, max_length, top_p, temperature, do_sample)
		st.text(prediction)
