#%% Import libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import siamese_network as SN
from PIL import Image
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import helper
import requests
from shutil import copy
import shutil

# path to all identified individuals
RIGHT_INDIVIDUAL_DATA_PATH = "./identified_individuals/right/"
LEFT_INDIVIDUAL_DATA_PATH = "./identified_individuals/left/"
TOP_INDIVIDUAL_DATA_PATH = "./identified_individuals/top/"

# list containing the names of the entries individuals in the specific side
RIGHT_INDIVIDUAL_LIST = os.listdir(RIGHT_INDIVIDUAL_DATA_PATH)
LEFT_INDIVIDUAL_LIST = os.listdir(LEFT_INDIVIDUAL_DATA_PATH)
TOP_INDIVIDUAL_LIST = os.listdir(TOP_INDIVIDUAL_DATA_PATH)

# path to all models
LEFT_MODEL_PATH = './models/left_image_proc_training_gray_models/'
RIGHT_MODEL_PATH = './models/right_image_proc_training_gray_models/'
TOP_MODEL_PATH = './models/top_image_proc_training_gray_models/'
MODEL_NAME = 'siamese-face-model.h5'

DOWNLOADS_PATH = 'instance/'
TEMP_CROPPED_PATH = 'temp_cropped/'
CROPPED_PATH = 'cropped_images/'
IDENTIFIED_INDIVIDUALS_PATH = 'identified_individuals/'

# run application
app = Flask(__name__)
CORS(app)

global num_identifiers
num_identifiers = 3

# load model
def load_my_models(model_path):
    print('Loading the appropriate model')
    fit_model_path = os.path.join(model_path, MODEL_NAME)
    global model
    model = load_model(fit_model_path, custom_objects={'contrastive_loss': SN.contrastive_loss})
    print('My model {} was loaded.....'.format(fit_model_path))    

# basic configratuions
UPLOAD_FOLDER                       =   './static/'
ALLOWED_EXTENSIONS                  =   set(['png', 'bmp', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER']         =   UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']    =   1 * 1200 * 1200

def check_sides(photo):
    # check side of specific image
    rightSide = photo['RightSide']
    leftSide = photo['LeftSide']
    topSide = photo['TopSide']
    
    if rightSide:
        side = 'right'
    elif leftSide:
        side = 'left'
    elif topSide:
        side = 'top'
    else: return 'not found available side'
    
    print('The checked side is:', side)
    return side

# app route - get json contain url's & BB, download images, cropping the img according the BB
# make image processing on cropped img & send the CROPPED_PATH to identification func
@app.route('/identifyPhotos', methods=['POST'])
def identifyPhotos():
    #delete stucked files
    if os.path.exists(TEMP_CROPPED_PATH):
        shutil.rmtree(TEMP_CROPPED_PATH)
    if request.method == 'POST':
        json_info = request.get_json(force=True)
        boundingBoxes_list = json_info['boundingBoxes']
        photos_list = json_info['photos']
        print('Photos_list', photos_list)
        bb_list = json_info['boundingBoxes']
        result = []
        all_identifications = {}
        all_identifications_from_sides = []
        
        # download all images from URL's (src)
        if not os.path.exists(DOWNLOADS_PATH):
            os.makedirs(DOWNLOADS_PATH)
            print('The folder {} was created'.format(DOWNLOADS_PATH))
        
        for photo in photos_list:
            r = requests.get(photo['src'])            
            with app.open_instance_resource('{}'.format(photo['value']), 'wb') as f:
                f.write(r.content)

        for bb in bb_list:
            photo_id = bb['PhotoID']
            
            # looking for the side of this image for tagging the crop image in the same side
            for photo in photos_list:
                if photo['value'] == photo_id:
                    side = check_sides(photo)
                    if side == 'Not found available side':
                        return side + ' for identification, Please select side for image {}'.format(photo_id)
                    
            # looking for the saved image belonging to this photo_id (BB)
            for file in os.listdir(DOWNLOADS_PATH):
                if file == photo_id:
                    img = cv2.imread(os.path.join(DOWNLOADS_PATH, file))

                    w = bb['Width']
                    h = bb['Height']
                    x = bb['Left_x']
                    y = bb['Top_y']
                                     
                    # cropped the image according to tha BB
                    cropped = img[y:y+h, x:x+w]
                    
                    if not os.path.exists(TEMP_CROPPED_PATH + '{}'.format(side)):
                        os.makedirs(TEMP_CROPPED_PATH + '{}'.format(side))
                        print('The folder {} was created'.format(TEMP_CROPPED_PATH + '{}'.format(side)))
                    
                    # save the cropped image
                    cv2.imwrite('{}{}/{}'.format(TEMP_CROPPED_PATH, side, photo_id), cropped)

                    print('The photo {} wad cropped & saved in {}{}'.format(photo_id, TEMP_CROPPED_PATH, side))
                            
                    # delete donloaded image (after saving the crop image)
                    os.remove(DOWNLOADS_PATH + file)
                    
        for side in os.listdir(TEMP_CROPPED_PATH):
            print('Load model and identify all images for {} side'.format(side))
            # send all cropped images to imageProcessing func for extract spotts
            image_proc = imageProcessing(TEMP_CROPPED_PATH + side)
            if image_proc == 'All cropped images have been image processed':
                
             #selecting the appropriate model
                print('Selecting the appropriate model')
                # right side model
                if side == 'right':
                    model_path =  RIGHT_MODEL_PATH
                    print('Chosen model_path: ',model_path)
                
                # left side model
                elif side == 'left':
                    model_path = LEFT_MODEL_PATH
                    print('Chosen model_path: ',model_path)
                
                # top side model
                elif side == 'top':
                    model_path = TOP_MODEL_PATH
                    print('Chosen model_path: ',model_path)
            
                # loading the appropriate model
                load_my_models(model_path)

                all_identifications = identifications(TEMP_CROPPED_PATH + '{}'.format(side), photos_list, side)
                
                # append all identification of specific side, for create list of all sides identification
                all_identifications_from_sides.append(all_identifications)
            else: return 'Problem with image proccesing, try again'
        
        # delete the folder TEMP_CROPPED_PATH with all the files inside it
        shutil.rmtree(TEMP_CROPPED_PATH)
        
    return jsonify(all_identifications_from_sides)


# image processing to extract the spotts of the Bluespotted
def imageProcessing(input_path):
    print('Start image processing')
    for filename in os.listdir(input_path):
        img = cv2.imread(os.path.join(input_path, filename))
        file_name, ext = os.path.splitext(filename)
        alpha = 0.5  # Contrast control
        beta = 70  # Brightness control
        image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_lower = np.array([41,57,78])
        hsv_upper = np.array([145,255,255])
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        name = "./data/%s.jpg" % file_name
        cv2.imwrite('{}/{}.jpg'.format(input_path, file_name), mask)

    return 'All cropped images have been image processed'

    
# identifications func: get path to cropped images of specifec side, return list of all individual identifications
def identifications(side_cropped_path, photos_list, side): 
    print('Start identification for seleced side({})'.format(side))
    img_res = []
    _side = side
    if len(side_cropped_path) == 0:
        return("Error: No image file ")
    
    for img in os.listdir(side_cropped_path):
        image = os.path.join(side_cropped_path, img)
        original_file_name = img
        # copy the cropped images for reuse #diffrent dir
        copy(image, CROPPED_PATH)
        
        # extract faces
        print("Extracting a face")
        pixels = helper.extract_face(image, required_size=(320, 320))
        img = Image.fromarray(pixels, mode='RGB')
        tmp_filename = os.path.join(UPLOAD_FOLDER, 'tmp_rgb.jpg')
        img.save(tmp_filename)

        # convert to grayscale
        print("Converting the image")
        img = cv2.imread(tmp_filename, cv2.IMREAD_GRAYSCALE)

        target_path = os.path.join(UPLOAD_FOLDER, 'tmp_gray.jpg')
        cv2.imwrite(target_path, img) 

        # predict - send the query image and his side
        print("Predicting the face")
        ret_val = make_prediction(target_path, _side)
        data = {}
        individual_data = {}
        all_individuals_data = []
        for res in ret_val:
            individual_data['id'] = res[0]
            individual_data['image_name'] = res[1]
            all_individuals_data.append(individual_data.copy())
  
        data['Similar_individuals'] = all_individuals_data
            
        # get the original URL corresponding to the image
        for photo in photos_list:
            if photo['value'] == original_file_name:
                data['src'] = photo['src']
    print(data)
    return data
    
def make_prediction(file_path, side):
    print('Starting prediction for image {} from the side {}'.format(file_path, side))
    _side = side
    ref_image = helper.get_image_from_filename(file_path)    
    ref_image_trans = tf.transpose(ref_image, perm=[0, 2, 3, 1])
    
    # find the category
    results = []
    image_names = []
    cat = 0

    # navigate to the list of individuals in DB, of the same side
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
        individual_res = []

        # make list of all images of those individual 
        images_list = os.listdir(os.path.join(INDIVIDUAL_DATA_PATH, individual))

        # checks how many images are available for each individual
        img_count = len(images_list)
        
        for img in range(img_count):
            cur_image = helper.get_image(INDIVIDUAL_DATA_PATH, cat, img)   
            cur_image_trans = tf.transpose(cur_image, perm=[0, 2, 3, 1])
        
            # make prediction between 2 images and return Numpy array of predictions(distance)
            distance = model.predict([[ref_image_trans], [cur_image_trans]])[0][0]

            # array of the distances Between the query image and all the individual images
            individual_res.append(distance)
            
        # get the min distance from this individual
        min_val = min(individual_res)
        
        # get the index of the min distance from this individual
        min_index = individual_res.index(min_val)

        # get the image_name of the min disntance from this individual
        image_names.append(images_list[min_index])
        
        # append the min distance, to get list of distances between all individuals
        results.append(min_val)

        cat += 1
            
            
    # get the 'num_identifiers' indexes of the min distances    
    idx_min_dis = np.argsort(results)[:num_identifiers]
    
    # get the ID's of the 'num_identifiers' min distances from individuals_list 
    id_list = [INDIVIDUAL_LIST[num_identifiers] for num_identifiers in idx_min_dis]
    
    # get the image_names of the 'num_identifiers' min distances from image_names
    image_name_list = [image_names[num_identifiers] for num_identifiers in idx_min_dis]
    
    # concatenate id_list & image_name_list for matching each individual ID to its image_name
    res = list(zip(id_list, image_name_list))

    return res

# app route - get original image and individual_ID selected by the researcher
# tag the image according to the individual received
@app.route('/setIndividualIdentity', methods=['POST'])
def setIndividualIdentity():
    if request.method == 'POST':
        json_info = request.get_json(force=True)
        individual_ID =  json_info['individual_ID']
        original_img_name = json_info['value']
        side = check_sides(json_info)
        
        if side == 'Not found available side':
            return side + ' for tagging, Please select side for image {}'.format(original_img_name)
        
        ID_path = IDENTIFIED_INDIVIDUALS_PATH + side + '/{}'.format(individual_ID)
        
        # if it's a new individual_ID - create folder for him
        if not os.path.isdir(ID_path):
            os.mkdir(ID_path)
 
        if len(os.listdir(CROPPED_PATH)) != 0:
            # search the cropped img
            for cropped_img in os.listdir(CROPPED_PATH):
                if original_img_name == cropped_img:
                    src = CROPPED_PATH + '/' + cropped_img
                    img = Image.open(src)
                    img_resize = img.resize((320,320), Image.ANTIALIAS)
                    img_resize.save(CROPPED_PATH + '{}'.format(original_img_name), quality=200)
                    
                    # move the cropped img to his identified individuals folder
                    destination = shutil.move(os.path.join(CROPPED_PATH, cropped_img), os.path.join(ID_path, cropped_img))
                    return 'Image {} was successfully tagged as individual ID-{} on VM'.format(original_img_name, individual_ID)
                
    return "The directory {} is empty: there is no images to tag".format(CROPPED_PATH)

# main
if __name__ =='__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

