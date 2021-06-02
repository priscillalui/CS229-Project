import streamlit as st
from pathlib import Path
import os

import numpy as np
import PIL
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
# import clip

from inference import setup, single_inference
from miscc.utils import save_test_samples, images_to_numpy


app_formal_name = "StorybookGAN"

# Temporary variables
# TODO: Complete inference pipeline
device = "cuda"
desc_dir = '/home/ubuntu/StoryGAN/vist_dataset'
video_len = 5

# TODO: Make into One-time setup variables
@st.cache
def one_time_setup():
    gan_args = setup() # sets up StoryGAN
    return gan_args

# Start the app in wide-mode
st.set_page_config(
    layout="wide", page_title=app_formal_name, initial_sidebar_state="expanded"
)

# If the user has a custom story, enter it here
with st.beta_expander("Make your own storybook!"):
    sentences = st.text_area(
        "Input your story here, one line per sentence, five in total. Press [Control+Enter] to compute.",
        value="\n".join(["i went to a party today", "it was fun", "it was on a beach", "we drank juice", "i love my friends"])
    ).split("\n")

    # Run setup code
    gan_args = one_time_setup()
    # clip_model,_ = clip.load('ViT-B/32', device) # sets up CLIP


    if len(sentences) != 5:
        st.error("Please enter a valid input")
    else:
        st.info("Encoding your story...")
        # tokenized_descriptions = torch.cat([clip.tokenize(s) for s in sentences]).to(device)
        # with torch.no_grad():
        #     encoded_descriptions = clip_model.encode_text(tokenized_descriptions).float()

        # # Run the encoded story text through StoryGAN 
        # fake_imgs = single_inference(gan_args, encoded_descriptions)

        # Show the credits for each photo in an expandable sidebar
        st.markdown(f"## Your Storybook: \n")
        col_generator = st.beta_columns(video_len)
        for i, col in enumerate(col_generator):
            # images = fake_imgs.squeeze(0).transpose(0, 1)[i].squeeze(0)
            # images = images_to_numpy(images)
            # image = PIL.Image.fromarray(images)

            with col:
                st.image('/home/Priscilla/VIST/Test/images/106158.jpg', use_column_width='always')
                st.text(sentences[i])

        # Format
        st.sidebar.title("Storybook Illustrator")
        st.sidebar.markdown("-----------------------------------")
        st.sidebar.markdown(
            f"[{app_formal_name}](https://github.com/eunjeeSung/StoryGAN) "
            f"creates storybook illustrations from stories."
            f" The model was trained on a GANILLA-fied subset of the VIST dataset from Microsoft."
        )
        st.sidebar.markdown(
            "Made with 💙 by Eun Jee, Mira, and Priscilla for Stanford 2020-2021 Winter **CS236G: Generative Adversarial Network** final project"
        )

