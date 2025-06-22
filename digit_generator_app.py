import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import DigitGenerator

# Page title
st.title("🧠 Handwritten Digit Generator")

# Select digit
digit = st.number_input("🎯 Choose a digit (0–9):", min_value=0, max_value=9, step=1)

# Load the trained model
model = DigitGenerator()
model.load_state_dict(torch.load("generator.pth", map_location=torch.device('cpu')))
model.eval()

# Generate images when user clicks the button
if st.button("Generate Images"):
    st.subheader(f"Generated images of digit **{digit}**")

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        z = torch.randn(1, 100)  # random noise vector
        label = torch.zeros(1, 10)
        label[0][digit] = 1  # one-hot label

        with torch.no_grad():
            image = model(z, label).squeeze().numpy()

        axs[i].imshow(image, cmap='gray')
        axs[i].axis('off')

    st.pyplot(fig)

st.markdown("---")
st.caption("Built with ❤️ using PyTorch + Streamlit")
