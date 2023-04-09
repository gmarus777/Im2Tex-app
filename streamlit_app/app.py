import cv2
import numpy as np
from PIL import Image
import streamlit
import torch
from albumentations.augmentations.geometric.resize import Resize
import torch.nn.functional as F
from pathlib import Path
import sys
from load_model import load_weights

sys.path.append(str(Path(__file__).resolve().parent.parent))
from Tokenizer.Tokenizer import token_to_strings
from Tokenizer.transform import Image_Transforms

MAX_RATIO = 8
GOAL_HEIGHT = 128
max_H = 128
max_W = 1024

# Hosted on my personal account until I figure something else out
cloud_model_location = "1j-ECpr0PIVbJGeRKFeYozDq7M4urk7sP"








streamlit.set_page_config(page_title='LaTeX OCR Model',
                          layout="centered",
                          menu_items={
                              'Get Help': 'https://github.com/gmarus777/image-to-tex-OCR',
                              'Report a bug': "https://github.com/gmarus777/image-to-tex-OCR",
                              'About': "# LaTeX *OCR* Model"
                          }
                          )

streamlit.title('LaTeX OCR')
streamlit.markdown('Convert Math Formula Images to LaTeX Code.\n\nBased on the `image-to-tex-OCR` project. For more details  [![github](https://img.shields.io/badge/image--to--Tex--OCR-visit-a?style=social&logo=github)](https://github.com/gmarus777/image-to-tex-OCR)')

uploaded_image = streamlit.file_uploader(
    'Upload Image',
    type=['png', 'jpg'],
)


#@streamlit.cache
def load_model():
    path = load_weights()
    model = torch.jit.load(path)
    return model

model = load_model




if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    streamlit.image(image)
    image = np.asarray(image)



    h, w, c = image.shape
    ratio = w / h
    if ratio == 0:
        ratio = 1
    if ratio > MAX_RATIO:
        ratio = MAX_RATIO

    new_h = GOAL_HEIGHT
    new_w = int(new_h * ratio)
    image = Resize(interpolation=cv2.INTER_LINEAR, height=new_h, width=new_w, always_apply=True)(image=image)['image']

    image_tensor = Image_Transforms.test_transform_with_padding(image=np.array(image))['image']  # [:1]

    image_tensor = F.pad(image_tensor, (0, max_W - new_w, 0, max_H - new_h), value=0)



if streamlit.button('Convert'):
    if uploaded_image is not None and image_tensor is not None:
        files = {"file": uploaded_image.getvalue()}
        with streamlit.spinner('Converting Image to LaTeX'):

            prediction = model(image_tensor.unsqueeze(0))
            latex_code = token_to_strings(tokens=prediction)

            #Docker image
            #response = requests.post('http://0.0.0.0:8000/predict/', files={'file': uploaded_image.getvalue()})

            streamlit.title('Result')
            streamlit.markdown('LaTeX Code:')
            streamlit.code(latex_code, language='latex')
            streamlit.markdown('Rendered LaTeX:')
            streamlit.markdown(f'$\\displaystyle {latex_code}$')


    else:
        streamlit.error('Please upload an image.')





