#%% Import libraries
import os
import cv2
import numpy as np
import tensorflow as tf

from keras.models import load_model
import siamese_network as SN

from PIL import Image

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from time import gmtime, strftime
import helper

import time
from io import BytesIO
from datetime import timedelta

import requests
from urllib.parse import urlparse

from shutil import copy
import shutil

#Delete befor deployment  !!!!!!
from flask_cors import CORS #comment this on deployment

#path to all identified individuals
RIGHT_INDIVIDUAL_DATA_PATH = "./identified_individuals_testing/right/"
LEFT_INDIVIDUAL_DATA_PATH = "./identified_individuals_testing/left/"
TOP_INDIVIDUAL_DATA_PATH = "./identified_individuals_testing/top/"

#list containing the names of the entries individuals in the directory
RIGHT_INDIVIDUAL_LIST = os.listdir(RIGHT_INDIVIDUAL_DATA_PATH)
# print('RIGHT_INDIVIDUAL_LIST', RIGHT_INDIVIDUAL_LIST)
LEFT_INDIVIDUAL_LIST = os.listdir(LEFT_INDIVIDUAL_DATA_PATH)
# print('LEFT_INDIVIDUAL_LIST', LEFT_INDIVIDUAL_LIST)
TOP_INDIVIDUAL_LIST = os.listdir(TOP_INDIVIDUAL_DATA_PATH)
# print('TOP_INDIVIDUAL_LIST', TOP_INDIVIDUAL_LIST)

#path to all models
LEFT_MODEL_PATH = './models/left_image_proc_training_gray_models/'
RIGHT_MODEL_PATH = './models/right_image_proc_training_gray_models/'
TOP_MODEL_PATH = './models/top_image_proc_training_gray_models/'
MODEL_NAME = 'siamese-face-model.h5'

#path to downloaded images from DB
downloaded_images_path = 'instance/'

#path to temporary cropped images
temp_cropped_path = 'temp_cropped/'

#path to cropped images
cropped_path = 'cropped_images/'

#####################################################################
#path to identified individuals   ###################################
identified_individuals_path = 'identified_individuals_testing/'

#%% Run app
app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)

#Delete befor deployment!!!!!!
CORS(app) #comment this on deployment

#%% Load model
def load_my_models(model_path):
    print('Loading the appropriate model')
    fit_model_path = os.path.join(model_path, MODEL_NAME)
    global model
    model = load_model(fit_model_path,
                       custom_objects={'contrastive_loss': SN.contrastive_loss})
    print(' * My model {} was loaded.....'.format(fit_model_path))    

#%% Basic configratuions
UPLOAD_FOLDER                       =   './static/images/'
ALLOWED_EXTENSIONS                  =   set(['png', 'bmp', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER']         =   UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']    =   1 * 1200 * 1200


def check_sides(photo):
    #check side of specific image
    rightSide = True if photo['RightSide'] == 'TRUE' else False
    leftSide = True if photo['LeftSide'] == 'TRUE' else False
    topSide = True if photo['TopSide'] == 'TRUE' else False
    
    if rightSide:
        side = 'right'
    elif leftSide:
        side = 'left'
    elif topSide:
        side = 'top'
    else: return 'not found available side for identification'
    
    print('the checked side is:', side)
    return side


# %% app route - get json contain url's & BB, download images, cropping the img according the BB
# make image processing on cropped img & send the cropped_path to identification func
@app.route('/identifyPhotos', methods=['POST'])
def identifyPhotos():
    if request.method == 'POST':
        json_info = request.get_json(force=True)
        boundingBoxes_list = json_info['boundingBoxes']
        photos_list = json_info['photos']
        print('photos_list', photos_list)
        bb_list = json_info['boundingBoxes']
        result = []
        all_identifications = {}
        all_identifications_from_sides = []
        
        #download all images from URL's (src)
        for photo in photos_list:
            print('download photo', photo)
            r = requests.get(photo['src'])            
            with app.open_instance_resource('{}'.format(photo['value']), 'wb') as f:
                f.write(r.content)

        for bb in bb_list:
            photo_id = bb['PhotoID']
            print('the checked photo_id:', photo_id)
            
            #looking for the side of this image for tagging the crop image in the same side
            for photo in photos_list:
                if photo['value'] == photo_id:
                    print('the photo that found is:', photo['value'])
                    side = check_sides(photo)
                    
            #Looking for the saved image belonging to this photo_id (BB)
            for file in os.listdir(downloaded_images_path):
                if file == photo_id:
                    print('the saved file to cropped was found:', file )
                    img = cv2.imread(os.path.join(downloaded_images_path, file))
                    w = bb['Width']
                    h = bb['Height']
                    x = bb['Left_x']
                    y = bb['Top_y']
                                     
                    #cropped the image according to tha BB
                    cropped = img[y:y+h, x:x+w]
                    
                    if not os.path.exists(temp_cropped_path + '{}'.format(side)):
                        os.makedirs(temp_cropped_path + '{}'.format(side))
                        print('the folder {} was created'.format(temp_cropped_path + '{}'.format(side)))
                    
                    #save the cropped image
                    cv2.imwrite('{}{}/{}'.format(temp_cropped_path, side, photo_id), cropped)
                    print('the photo {} wad cropped & saved in {}{}'.format(photo_id, temp_cropped_path, side))

        for side in os.listdir(temp_cropped_path):
            print('load model and identify all images for {} side'.format(side))
            #send all cropped images to imageProcessing func for extract spotts
            image_proc = imageProcessing(temp_cropped_path + side)
            if image_proc == 'All cropped images have been image processed':
                
              #selecting the appropriate model
                print('selecting the appropriate model')
                #right side model
                if side == 'right':
                    model_path =  RIGHT_MODEL_PATH
                    print('chosen model_path: ',model_path)
                
                #left side model
                elif side == 'left':
                    model_path = LEFT_MODEL_PATH
                    print('chosen model_path: ',model_path)
                
                #top side model
                elif side == 'top':
                    model_path = TOP_MODEL_PATH
                    print('chosen model_path: ',model_path)
            
                #Loading the appropriate model
                load_my_models(model_path)

                all_identifications = identifications(temp_cropped_path + '{}'.format(side), photos_list, side)
                
                #append all identification of specific side, for create list of all sides identification
                all_identifications_from_sides.append(all_identifications)
            else: return 'problem with image proccesing, try again'
        
        #delete the folder temp_cropped_path with all the files inside it
        shutil.rmtree(temp_cropped_path)
    return jsonify(all_identifications_from_sides)
    
#Image processing to extract the spotts of the bluespotted    
def imageProcessing(input_path):
    print('start image processing')
    for filename in os.listdir(input_path):
        print('file for processing:', filename)
        img = cv2.imread(os.path.join(input_path, filename))
        file_name, ext = os.path.splitext(filename)
        alpha = 0.5  # Contrast control (1.0-3.0)
        beta = 70  # Brightness control (0-100)
        image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_lower = np.array([41,57,78])
        hsv_upper = np.array([145,255,255])
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        name = "./data/%s.jpg" % file_name
        cv2.imwrite('{}/{}.jpg'.format(input_path, file_name), mask)
    cv2.waitKey()
    return 'All cropped images have been image processed'

    
# identifications func: gets path to cropped images of specifec side, return list of all individual identifications
def identifications(some_cropped_path, photos_list, side): 
    print('start identification for seleced side({})'.format(side))
    img_res = []
    _side = side
    if len(some_cropped_path) == 0:
        return("Error: No image file ")
    
    for img in os.listdir(some_cropped_path):
        image = os.path.join(some_cropped_path, img)
        original_file_name = img
        #copy the cropped images for reuse #diffrent dir
        copy(image, cropped_path)
        # extract faces
        print("....... Now extracting a face .......")
        pixels = helper.extract_face(image, required_size=(320, 320))
        img = Image.fromarray(pixels, mode='RGB')
        tmp_filename = os.path.join(UPLOAD_FOLDER, 'tmp_rgb.jpg')
        img.save(tmp_filename)

        # convert to grayscale
        print("....... Now converting the image .......")
        img = cv2.imread(tmp_filename, cv2.IMREAD_GRAYSCALE)
        target_path = os.path.join(UPLOAD_FOLDER, 'tmp_gray.jpg')
        cv2.imwrite(target_path, img) 

        # predict
        print("....... Now predicting the face .......")
        #send to prediction the query image and his side
        ret_val = make_prediction(target_path, _side)
        data = {}
        data["individuals_ID"] = ret_val
        
        #get the original URL corresponding to the image
        for photo in photos_list:
            if photo['value'] == original_file_name:
                data['src'] = photo['src']
        
        #Merge the original URL with the identified individuals_ID
        img_res.append(data)
    
    return img_res
    
#%% function: make_prediction()
def make_prediction(file_path, side):
    print(' * Starting prediction for image {} from the side {}.....'.format(file_path, side))
    _side = side
    ref_image = helper.get_image_from_filename(file_path)    
    ref_image_trans = tf.transpose(ref_image, perm=[0, 2, 3, 1])
    
    # find the category
    individual_res = []
    results = []
    cat = 0
    k = 3

    # Navigate to the list of individuals in DB, of the same side
    if side == 'right':
        INDIVIDUAL_LIST = RIGHT_INDIVIDUAL_LIST
        INDIVIDUAL_DATA_PATH = RIGHT_INDIVIDUAL_DATA_PATH
    if side == 'left':
        INDIVIDUAL_LIST = LEFT_INDIVIDUAL_LIST
        INDIVIDUAL_DATA_PATH = LEFT_INDIVIDUAL_DATA_PATH
    if side == 'top':
        INDIVIDUAL_LIST = TOP_INDIVIDUAL_LIST
        INDIVIDUAL_DATA_PATH = TOP_INDIVIDUAL_DATA_PATH
    
    
    for individual in INDIVIDUAL_LIST:        
        
        #make list of all images of those individual 
        images_list = os.listdir(os.path.join(INDIVIDUAL_DATA_PATH, individual))

        #Checks how many images are available for each individual
        img_count = len(images_list)
        
        for img in range(img_count):
            cur_image = helper.get_image(INDIVIDUAL_DATA_PATH, cat, img)   
            cur_image_trans = tf.transpose(cur_image, perm=[0, 2, 3, 1])
        
            # Make prediction between 2 images and return Numpy array of predictions(distance)
            distance = model.predict([[ref_image_trans], [cur_image_trans]])[0][0]

            # Array of the distances between 2 images
            individual_res.append(distance)
            
        #get the min distance from this class(individuals)
        min_val = min(individual_res)
        
        #append the min distances, to get list of distances between all individuals
        results.append(min_val)

        cat += 1
            
            
    # converting list to array
    res_arr = np.array(results)
    print('res_arr', res_arr)
    
    # Get the K indexes of the min distances    
    idx_min_dis = np.argsort(res_arr)[:k]
    print('idx_min_dis:', idx_min_dis)
    
    #get the ID's of the k min distances from individuals_list 
    id_list = [INDIVIDUAL_LIST[k] for k in idx_min_dis]
    print('id_list:', id_list)

    return id_list

# %% app route - get original image and individual_ID selected by the researcher
# tag the image according to the individual received
@app.route('/setIndividualIdentity', methods=['POST'])
def setIndividualIdentity():
    if request.method == 'POST':
        json_info = request.get_json(force=True)
        individual_ID =  json_info['individual_ID']
        original_img_name = json_info['value']
        side = check_sides(json_info)

        ID_path = identified_individuals_path + side + '/{}'.format(individual_ID)
        
        #if it's a new individual_ID - create folder for him
        if not os.path.isdir(ID_path):
            os.mkdir(ID_path)
 
        if len(os.listdir(cropped_path)) != 0:
            #search the cropped img
            for cropped_img in os.listdir(cropped_path):
                if original_img_name == cropped_img:
                    src = cropped_path + '/' + cropped_img
                    img = Image.open(src)
                    img_resize = img.resize((320,320), Image.ANTIALIAS)
                    #img_resize.save(original_img_name, quality=200)
                    #move the cropped img to his identified individuals folder
                    destination = shutil.move(src, ID_path)
                    return 'image {} was successfully tagged as individual ID-{} on VM'.format(original_img_name, individual_ID)
                
    return "The directory {} is empty: there is no images to tag".format(cropped_path)


#%% Main
if __name__ == '__main__':
    #app.run(debug=False, threaded=False)
    app.run(debug=True, host="0.0.0.0", port =5000)