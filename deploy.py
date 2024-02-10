import gradio as gr
import joblib
import torch
import numpy as  np
from PIL import Image
from mnist import classifying_model

model = joblib.load('model.pkl')
def detect(image):
    im = image.getchannel(1).resize([28, 28])
    img = np.array(im)
    img = img.astype('float32')
    np.resize(img, [28, 28]).reshape(1,1,28,28)
    img = torch.Tensor(img)
    batch_image = img.repeat(2, 1, 1, 1)
    pred = model(batch_image)
    _, prediction = torch.max(pred, 1)
    return im, prediction.numpy()[0]
    # return ' '.join([i for i in dir(img) if not i.startswith('_')])
interface = gr.Interface(fn = detect, inputs=gr.Image(type = 'pil'), outputs = ['image', 'text'])
interface.launch()


