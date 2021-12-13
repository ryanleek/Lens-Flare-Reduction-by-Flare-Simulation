import pathlib
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize

from unet import UNet
from transformations import normalize_01, re_normalize


# root directory
root = pathlib.Path.cwd() / 'dataset'
def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames

# input and target files
images_names = get_filenames_of_path(root)

# read images and store them in memory
images = [imread(img_name) for img_name in images_names]

# Resize images and targets
images_res = [resize(img, (256, 256, 3)) for img in images]
resize_kwargs = {'order': 0, 'anti_aliasing': False, 'preserve_range': True}

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    torch.device('cpu')

# model
model = UNet(in_channels=3,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)


model_name = 'flare_model.pt'
model_weights = torch.load(pathlib.Path.cwd() / model_name)

model.load_state_dict(model_weights)

def predict(img,
            model,
            preprocess,
            postprocess,
            device,
            ):
    model.eval()
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network

    out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
    result = postprocess(out_softmax)  # postprocess outputs

    return result

# preprocess function
def preprocess(img: np.ndarray):
    img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
    img = normalize_01(img)  # linear scaling to range [0-1]
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    return img


# postprocess function
def postprocess(img: torch.tensor):
    img = torch.argmax(img, dim=1)  # perform argmax to generate 1 channel
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    img = re_normalize(img)  # scale it to the range [0-255]
    return img

# predict the segmentation maps 
output = [predict(img, model, preprocess, postprocess, device) for img in images_res]

#plt.imshow(output[3])
#plt.imsave('4_1t.jpg', output[3], cmap = cm.gray)


##image inpainting

#img = cv2.imread('origin.jpg')
#mask = cv2.imread('mask.jpg', 0)

#paint = cv2.inpaint(img, mask, 100, cv2.INPAINT_TELEA)

#cv2.imwrite('output.jpg', paint)