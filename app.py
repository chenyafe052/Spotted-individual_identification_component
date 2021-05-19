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

#%% Basic configurations
INDIVIDUAL_DATA_PATH = "./dataset/individuals/aligned_after_image_processing_gray/"

#list containing the names of the entries individuals in the directory
INDIVIDUAL_LIST = os.listdir(INDIVIDUAL_DATA_PATH)

print(INDIVIDUAL_LIST) 

MODEL_PATH = './aligned_after_image_processing_gray_models/'
MODEL_NAME = 'siamese-face-model.h5'

#path to downloaded images
downloaded_images_path = 'instance/'

#path to temporary cropped images
temp_cropped_path = 'temp_cropped/'

#path to cropped images
cropped_path = 'cropped_images/'

#path to identified individuals
identified_individuals_path = 'identified_individuals/'

#%% Run app
app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)

#Delete befor deployment!!!!!!
CORS(app) #comment this on deployment

#%% Load model
def load_my_model():
    model_path = os.path.join(MODEL_PATH, MODEL_NAME)
    global model
    model = load_model(model_path,
                       custom_objects={'contrastive_loss': SN.contrastive_loss})
    print(' * My model loaded.....')


#%% Basic configratuions
UPLOAD_FOLDER                       =   './static/images/'
ALLOWED_EXTENSIONS                  =   set(['png', 'bmp', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER']         =   UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']    =   1 * 1200 * 1200


# %% app route - get json contain url's & BB, download images, cropping the img according the BB
# send the cropped_path to identification func
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
        
        #download all images from URL's (src)
        for photo in photos_list:
            r = requests.get(photo['src'])
            with app.open_instance_resource('{}'.format(photo['value']), 'wb') as f:
                f.write(r.content)
                
        for bb in bb_list:
            photo_id = bb['PhotoID']
            
            #Look for the saved image belonging to photo_id (BB)
            for file in os.listdir(downloaded_images_path):
                if file == photo_id:
                    img = cv2.imread(os.path.join(downloaded_images_path, file))
                    w = bb['Width']
                    h = bb['Height']
                    x = bb['Left_x']
                    y = bb['Top_y']
                                     
                    #cropped the image according to tha BB
                    cropped = img[y:y+h, x:x+w]
                    
                    if not os.path.exists(temp_cropped_path):
                        os.makedirs(temp_cropped_path)
                    
                    #save the cropped image
                    cv2.imwrite('{}{}'.format(temp_cropped_path, photo_id), cropped)
                    print('the photo {} wad cropped & saved in cropped_BB'.format(photo_id))
#                     result_string = 'the photo {} wad cropped & saved in cropped_BB'.format(photo_id)
#                     result.append(result_string)
                    
        all_identifications = identifications(temp_cropped_path, photos_list)
        
        #delete the folder temp_cropped_path with all the files inside it
        shutil.rmtree(temp_cropped_path)
    return jsonify(all_identifications)
    
    
    
# identifications func: gets path to represent cropped images, return list of all individual identifications
def identifications(represent_cropped_path, photos_list): 
    img_res = []
    if len(represent_cropped_path) == 0:
        return("Error: No image file ")
    
    for img in os.listdir(represent_cropped_path):
        image = os.path.join(represent_cropped_path, img)
        original_file_name = img
        #copy the cropped images for reuse
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
        ret_val = make_prediction(target_path)
        data = {}
        data["individuals_ID"] = ret_val.tolist()
        #data["original_image"] = image
        
        #get the original URL corresponding to the image
        for photo in photos_list:
            if photo['value'] == original_file_name:
                data['src'] = photo['src']
        
        #Merge the original URL with the identified individuals_ID
        img_res.append(data)
    
    return img_res
    
#%% function: make_prediction()
def make_prediction(file_path):
    print(' * Starting prediction.....')
    ref_image = helper.get_image_from_filename(file_path)
    ref_image_trans = tf.transpose(ref_image, perm=[0, 2, 3, 1])
    
    # find the category
    results = []
    cat = 0
    num_of_val = 3
    
    # for every individual from Database
    for individual in INDIVIDUAL_LIST:
        #make list of all images of those individual 
        images_list = os.listdir(os.path.join(INDIVIDUAL_DATA_PATH, individual))
        
        #Checks how many images are available for each individual
        idx_list = len(images_list)
        
        for img in range(idx_list):
            cur_image = helper.get_image(INDIVIDUAL_DATA_PATH, cat, img)   
            cur_image_trans = tf.transpose(cur_image, perm=[0, 2, 3, 1])
        
            # Make prediction between 2 images and return Numpy array of predictions(distance)
            distance = model.predict([[ref_image_trans], [cur_image_trans]])[0][0]

            # Array of the distances between 2 images
            results.append(distance)
        cat += 1
            
    # converting list to array
    arr = np.array(results)
    
    # Calculates the average distances for all individual images
    avg_arr = np.mean(arr.reshape(-1, 4), axis=1)

    #Sort the individual array according to their values
    idx = np.argpartition(avg_arr, num_of_val)

    #return the individual_ID of the num_of_val smallest values
    num_of_val_smallest_values = idx[:3]

    return num_of_val_smallest_values


# %% app route - get original image and individual_ID selected by the researcher
# tag the image according to the individual received
@app.route('/setIndividualIdentity', methods=['POST'])
def setIndividualIdentity():
    if request.method == 'POST':
        json_info = request.get_json(force=True)
        individual_ID =  json_info['individual_ID']
        original_img_name = json_info['value']
        
        new_ID_path = identified_individuals_path + '/{}'.format(individual_ID)
        
        #if it is a new individual_ID - create folder for him
        if not os.path.isdir(new_ID_path):
            os.mkdir(new_ID_path) 
 
        if len(os.listdir(cropped_path)) != 0:
            #search the cropped img
            for cropped_img in os.listdir(cropped_path):
                if original_img_name == cropped_img:
                    src = cropped_path + '/' + cropped_img
                    #move the cropped img to his identified individuals folder
                    destination = shutil.move(src, new_ID_path)
                    return 'image {} was successfully tagged as individual ID-{} on VM'.format(original_img_name, individual_ID)
                
    return "The directory {} is empty".format(cropped_path)
        
    

# #%% app route - Upload image to the individual identification component
# @app.route('/identification', methods = ['GET', 'POST'])
# def uploader():
#     data = {}
#     if request.method == 'POST':
#         f = request.files['file']
#         filename_saved = secure_filename(f.filename)
       
#         # if no image file selected, stay upload.html page
#         if len(filename_saved) == 0:
#             return("Error: No image file ")

#         # extract file extension
#         file_ext = filename_saved.split('.')[-1]

#         #filename_saved = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
#         #filename_saved = "%s.%s" % (filename_saved, file_ext)
#         filename_saved = os.path.join(UPLOAD_FOLDER, filename_saved)

#         if os.path.isfile(filename_saved):
#             os.remove(filename_saved)

#         f.save(filename_saved)

#         # extract faces
#         print("....... Now extracting a face .......")
#         pixels = helper.extract_face(filename_saved, required_size=(320, 320))
#         img = Image.fromarray(pixels, mode='RGB')
#         tmp_filename = os.path.join(UPLOAD_FOLDER, 'tmp_rgb.jpg')
#         img.save(tmp_filename)

#         # convert to grayscale
#         print("....... Now converting the image .......")
#         img = cv2.imread(tmp_filename, cv2.IMREAD_GRAYSCALE)
#         target_path = os.path.join(UPLOAD_FOLDER, 'tmp_gray.jpg')
#         cv2.imwrite(target_path, img) 

#         # predict
#         print("....... Now predicting the face .......")
#         ret_val = make_prediction(target_path)
# #         ret_val = "Predicted: %s" % ret_val
#         data["result_id"] = ret_val
#         data["filename"] = filename_saved
#         res = jsonify(data)
#         return res    
    

# #Get MULTI url's of images, download & save them
# @app.route('/geturl', methods=['GET'])
# def uploadurl():
#     if request.method == 'GET':
#         # user provides url's in query string
#         url_list = request.args.getlist('url')
#         print('url_list: ', url_list)
#         image_name_list=[]
#         urls_list=[]
#         for url in url_list:
#             print('url', url)
            
#             r = requests.get(url)
#             image_name = urlparse(url).path.split('/')[-1]
            
#             # write to file in the app's instance folder
#             with app.open_instance_resource('{}'.format(image_name), 'wb') as f:
#                 f.write(r.content)
           
#             image_name_list.append(image_name)
#             urls_list.append(url)
#     all_files = jsonify(image_name_list)
#     all_urls = jsonify(urls_list)
#     return "the images {} from url's {} was saved".format(image_name_list, url_list)


#%% Main
if __name__ == '__main__':
    load_my_model()
    #app.run(debug=False, threaded=False)
    app.run(debug=True, host="0.0.0.0")