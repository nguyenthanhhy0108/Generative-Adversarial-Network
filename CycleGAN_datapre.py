import cv2
import os
from tqdm import tqdm
from PIL import Image

folder = "D:\\Machine Learning\\GAN\\day"
for image_name in tqdm(os.listdir(folder)):
    fullpath = os.path.join(folder, image_name)
    img = cv2.imread(fullpath)
    img = cv2.resize(img, (360, 360))
    new_folder = "D:\\Machine Learning\\GAN\\new_day"
    fullpath = os.path.join(new_folder, image_name)
    cv2.imwrite(fullpath, img)

folder = "D:\\Machine Learning\\GAN\\night"
for image_name in tqdm(os.listdir(folder)):
    fullpath = os.path.join(folder, image_name)
    img = cv2.imread(fullpath)
    img = cv2.resize(img, (360, 360))
    new_folder = "D:\\Machine Learning\\GAN\\new_night"
    fullpath = os.path.join(new_folder, image_name)
    cv2.imwrite(fullpath, img)