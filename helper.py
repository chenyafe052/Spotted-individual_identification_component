#%% Import libraries
import os
import cv2
from PIL import Image
from numpy import asarray
import matplotlib.image as mpimg


def read_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    return img


def get_image(data_path, category, index):
    pathlist = os.listdir(data_path)
    if category > len(pathlist):
        return None
    
    cur_path = os.path.join(data_path, pathlist[category])
    
    filelist = os.listdir(cur_path)
    if index > len(filelist):
        return None
    
    filepath = os.path.join(cur_path, filelist[index])
    
    image = read_image(filepath)
    size = 2
    image = image[::size, ::size]
    image = image.reshape(1, 1, image.shape[0], image.shape[1])
    image = image / 255
    
    return image


def get_image_from_filename(img_filename):
    image = read_image(img_filename)
    size = 2
    image = image[::size, ::size]
    image = image.reshape(1, 1, image.shape[0], image.shape[1])
    image = image / 255
    
    return image


def extract_face(filename, required_size=(600, 600)):
    # load image from file
    image = Image.open(filename)

    # convert to RGB, if needed
    image = image.convert('RGB')

    # convert to array
    pixels = asarray(image)
    image = image.resize(required_size)
    face_array = asarray(image)

    return face_array