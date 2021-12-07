from PIL import Image
import numpy as np
import glob
import cv2
import os
import cmapy

path = 'dataset/origin/'

for filename in os.listdir(path):
  img = Image.open(path+filename)

  h, w = img.height, img.width
  img = np.array(img)

  overlay = img.copy()
  target = np.zeros((h,w))

  #max_r = min(h,w)//6
  alpha = np.random.randint(10,25)/100

  for _ in range(np.random.randint(1,10)):
    x = np.random.randint(30,w-30)
    y = np.random.randint(30,h-30)
    center = (x, y)
    l = np.random.randint(15,50)
    rgb = cmapy.color('viridis', np.random.randint(0,256), rgb_order=True)

    ellipse_float = (center, (l, (np.random.randint(50,90)/100)*l), np.random.randint(-10,11))
    cv2.ellipse(overlay, ellipse_float, rgb, -1)
    cv2.ellipse(target, ellipse_float, 1, -1)

  img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
  img = Image.fromarray(img)
  target = Image.fromarray(np.uint8(target)*255)

  if img.mode in ("RGBA", "P"):
    img = img.convert("RGB")

  img.save('dataset/input/'+filename)
  target.save('dataset/target/'+filename)