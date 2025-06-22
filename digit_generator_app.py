import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import DigitGenerator

model = DigitGenerator()
model.load_state_dict(torch.load("generator.pth", map_location=torch.device('cpu')))
model.eval()

st.title("Handwritten Digit Generator (0–9)")
digit = st.number_input("Select a digit (0-9)", min_value=0, max_value=9, step=1)

if st.button("Generate Images"):
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        z = torch.randn(1, 100)
        label = torch.zeros(1, 10)
        label[0][digit] = 1
        with torch.no_grad():
            image = model(z, label).squeeze().numpy()
        axs[i].imshow(image, cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)