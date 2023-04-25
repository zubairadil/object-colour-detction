import os
import urllib
import imageio
import cv2
import numpy as np
import tarfile
import shutil
import onnx
import onnxruntime

from google_drive_downloader import GoogleDriveDownloader as gdd
from imread_from_url import imread_from_url


root_dir = "/home/hair_colour/"

model_path = root_dir+"models/hair_segmentation.onnx"

class HairSegmentation():

    def __init__(self, webcam_width, webcam_height):

        # Initialize model
        self.model = self.initialize_model()

    def __call__(self, image):

        return self.segment_hair(image)

    def initialize_model(self):

        # Create interpreter for the model
        self.session = onnxruntime.InferenceSession(model_path)

        # Get model info
        self.getModel_input_details()
        self.getModel_output_details()

    def segment_hair(self, image):

        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        # Process output data
        hair_mask = self.process_output(outputs)

        return hair_mask

    def prepare_input(self, image):

        self.img_height, self.img_width, self.img_channels = image.shape
        
        input_image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image  = cv2.resize(input_image , (self.input_width,self.input_height))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_image  = (input_image  / 255 - mean) / std

        input_image = input_image.transpose(2, 0, 1)
        input_tensor = input_image[np.newaxis,:,:,:]   

        return input_tensor.astype(np.float32)

    def inference(self, input_tensor):
        return self.session.run(self.output_names, {self.input_name: input_tensor})

    def process_output(self, outputs):  

        hair_mask = np.squeeze(outputs[0])
        hair_mask = hair_mask.transpose(1, 2, 0)
        hair_mask = hair_mask[:,:,2]
        hair_mask = cv2.resize(hair_mask, (self.img_width,self.img_height))

        return np.round(hair_mask).astype(np.uint8)

    def get_output_tensor(self, index):

        tensor = np.squeeze(self.interpreter.get_tensor(self.output_details[index]['index']))
        return tensor

    def getModel_input_details(self):

        self.input_name = self.session.get_inputs()[0].name

        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def getModel_output_details(self):

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[0].name]

        self.output_shape = model_outputs[0].shape
        self.output_height = self.output_shape[2]
        self.output_width = self.output_shape[3]

def find_contours_rectangle(mask):

    contours,hierarchy = cv2.findContours(mask*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        min_left = mask.shape[1]
        min_top  = mask.shape[0]
        max_right = 0 
        max_bottom = 0 
        for contour in contours:
            left, top, rect_width, rect_height = cv2.boundingRect(contour)
            bottom = top + rect_height
            right = left + rect_width

            min_left = min([min_left,left])
            min_top = min([min_top,top])
            max_right = max([max_right,right])
            max_bottom = max([max_bottom,bottom])

        contour_rectangle = [min_left, min_top, max_right, max_bottom]
    else:
        contour_rectangle = [0, 0, mask.shape[1], mask.shape[0]]

    return contour_rectangle


