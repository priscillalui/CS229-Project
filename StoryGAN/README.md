# Once Upon a GAN: StoryGAN trained on Visual Storytelling Dataset  
# Stanford CS236G Final Project

This repo builds upon the original StoryGAN repo. Novel contributions will include:
- A model trained on [VIST](https://visionandlanguage.net/VIST/), a photorealistic dataset containing "81,743 unique photos in 20,211 sequences". The original StoryGAN authors only showed results from training on a cartoon and clip art dataset.
- No more requirement for the entire story upfront. A user can start their storybook journey with just a single sentence.
- Style transfer feature, powered by GANILLA, to turn photorealistic outputs to beautiful illustrations.

## Note to TA:
To view the code that we authored, please refer to these files/directories:
- [vist_data.py](https://github.com/eunjeeSung/StoryGAN/blob/master/code/vist_data.py): Pytorch dataset loading.
- [main.py](https://github.com/eunjeeSung/StoryGAN/blob/master/code/main.py): Training pipeline.
- [/vist_dataset](https://github.com/eunjeeSung/StoryGAN/tree/master/vist_dataset): VIST annotations processing.

## Environment

```bash
conda env create -f environment.yml
```

## Data

### VIST Dataset (StoryGAN for Image Generation)

- Download annotations

Download the story-in-sequence annotations into `vist_dataset` folder from the [VIST Dataset page](http://visionandlanguage.net/VIST/dataset.html). Place `train.story-in-sequence.json` in `vist_dataset/train_annotation` folder and run the preprocess codes to encode the text annotations with CLIP.

- Download images

To download the images from the official [VIST Dataset page](http://visionandlanguage.net/VIST/dataset.html), please follow the directions [here](https://www.quora.com/How-do-I-download-a-very-large-file-from-Google-Drive/answer/Shane-F-Carr?ch=10&share=6509af0d&srid=hoGGk).

### Ganilla Dataset (CycleGAN for Stylization)

Please clone and refer to the directions [here](https://github.com/mirapartha/ganilla).

### miniGutenStories Dataset

Please download [images_grouped.zip](https://drive.google.com/file/d/10w-00iDJwdEumn61Z0m_ZVyEjeqSo2c3/view?usp=sharing).
Then unzip the file inside the `mini_guten_dataset` folder.


## Training

Train a StoryGAN model on the children's book data

1. Modify `code/cfg/vist.yml` for the VIST dataset.

2. Set `ENCODINGS_FILE`, `PICKLE_FILE`, and `LABEL_FILE` inside `code/vist_data.py` to the path to the sentence encoding files.
(`code/cfg/guten.yml` for the miniGutenStories dataset)
(*Under construction*)

3. Run the training code

```bash
python main.py \
--cfg ./cfg/vist.yml \
--img_dir /home/ubuntu/VIST/train \
--desc_path ../vist_dataset/train_annotation
```

4. Run tensorboard

```bash
tensorboard \
--log_dir=./runs \
--host=0.0.0.0 \
--port=<open port>
```


## Inference

1. Set `code/cfg/vist_test.yml` to use the Ganilla-processed VIST dataset.

2. Set `ENCODINGS_FILE`, `PICKLE_FILE`, and `LABEL_FILE` inside `code/vist_data.py` to the path to the sentence encoding files
(*Under construction*)

3. Run the inference code
```bash
python inference.py \
--cfg './cfg/vist_test.yml' \
--img_dir /home/ubuntu/VIST/ganilla_test \
--desc_path ~/StoryGAN/vist_dataset/test_annotation \
--output_dir ~/StoryGAN/output/test_batch
```

4. Output images are saved to `output/vist_StoryGAN/Test` by default.


## Streamlit

Please refer to the [Streamlit official documents](https://docs.streamlit.io/en/stable/index.html) for details.

1. Set `code/cfg/vist_streamlit.yml` to use the Ganilla-processed VIST dataset.

2. Set `ENCODINGS_FILE`, `PICKLE_FILE`, and `LABEL_FILE` inside `code/vist_data.py` to the path to the sentence encoding files
(*Under construction*)

3. Run the inference code
```bash
python inference.py \
--cfg './cfg/vist_streamlit.yml' \
--img_dir ~/VIST/ganilla_val \
--desc_path ~/StoryGAN/vist_dataset/sample_annotation
```

4. Currently output images will not be saved.


## TODO

- [ ] Evaluation code
- [ ] Pretrained models
- [ ] Refactoring the original main/util codes
  - [ ] Fix hardcoded values: clevr_data, guten_data
- [ ] Split util.py and loss.py
- [ ] Add encoding file paths to `config.py`

---

# StoryGAN: A Sequential Conditional GAN for Story Visualization (Python 3.7+, Pytorch 1.6)
This repository is still under construction. 

## Requirement:
Python 3.7+
Pytorch 1.6
Opencv-python (cv2)

## Configure File
/code/cfg/clevr.yml is the configure file for the model. This file contains the setup of the dimension of the features, maximum training epoches and etc.


## Run
To run the code on CLEVR-SV experiment:
```bash
python main_clevr.py
```

## Citation
```bash
@article{li2018storygan,
  title={StoryGAN: A Sequential Conditional GAN for Story Visualization},
  author={Li, Yitong and Gan, Zhe and Shen, Yelong and Liu, Jingjing and Cheng, Yu and Wu, Yuexin and Carin, Lawrence and Carlson, David and Gao, Jianfeng},
  journal={CVPR},
  year={2019}
}
```
