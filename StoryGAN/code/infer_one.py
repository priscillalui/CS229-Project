"""
Script that inferences on 5 sentences to generate 5 images.
"""

from pathlib import Path
import os

import numpy as np
import PIL
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import clip

from inference import setup, single_inference
from miscc.utils import save_test_samples, images_to_numpy


device = "cuda"
video_len = 5

gan_args = setup()

clip_model,_ = clip.load('ViT-B/32', device) # sets up CLIP

sentences = [
  "i went to a party today", 
  "it was fun", 
  "it was on a beach", 
  "we drank juice", 
  "i love my friends"
]

if len(sentences) != 5:
    print('Need 5 sentences')
else:
    tokenized_descriptions = torch.cat([clip.tokenize(s) for s in sentences]).to(device)
    with torch.no_grad():
        encoded_descriptions = clip_model.encode_text(tokenized_descriptions).float()

    # Run the encoded story text through StoryGAN 
    fake_imgs = single_inference(gan_args, encoded_descriptions)

    # Generate images
    vis = []
    for i in range(video_len):
        images = fake_imgs.squeeze(0).transpose(0, 1)[i].squeeze(0)
        images = images_to_numpy(images)
        # image = PIL.Image.fromarray(images)
        vis.append(images)
    horizontal_story = np.concatenate(vis, axis=1)
    horizontal_story = PIL.Image.fromarray(horizontal_story)
    horizontal_story.save('./infer_one_outputs/dii_epoch_19.jpg')