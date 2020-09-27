from .bounding_box_cone import BoundingBoxCone
from .models import Darknet
from .utils.utils import calculate_padding
from .utils import nms

import argparse
import os
from os.path import isfile, join
import random
import tempfile
import time
import copy
import multiprocessing
import subprocess
import shutil
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# from PIL import Image, ImageDraw
import torchvision
import warnings
from tqdm import tqdm
from timeit import default_timer as timer
import logging
warnings.filterwarnings("ignore")


class YoloV3Network:
    '''
    YoloV3 CNN Architecture for cone object detection.
    
    Vars:
        conf_thres - probability/confidence threshold,
        nms_thres - IoU threshold for non-maximum suppression,
        device,
        model - Darknet model,
    '''

    def __init__(self, weights_path, model_cfg, conf_thres = 0.8, nms_thres = 0.25, xy_loss = 2, wh_loss = 1.6, no_object_loss = 25, object_loss=0.1, vanilla_anchor = False):
        """
        Constructor.

        Args:
            weights_path (file path): pre trained weights file path.
            model_cfg ([cfg file]): YoloV3 architecture configuration file.
            conf_thres (float, optional): probability/confidence threshold. Defaults to 0.8.
            nms_thres (float, optional): IoU threshold for non-maximum suppression. Defaults to 0.25.
            xy_loss (int, optional): confidence loss for x and y. Defaults to 2.
            wh_loss (float, optional): confidence loss for width and height. Defaults to 1.6.
            no_object_loss (int, optional): confidence loss for background. Defaults to 25.
            object_loss (float, optional): confidence loss for foreground. Defaults to 0.1.
            vanilla_anchor (bool, optional): True or False, whether to use vanilla anchor boxes for training. Defaults to False.
        """
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        random.seed(0)
        torch.manual_seed(0)
        if cuda:
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        self.model = Darknet(config_path=model_cfg,xy_loss=xy_loss,wh_loss=wh_loss,no_object_loss=no_object_loss,object_loss=object_loss,vanilla_anchor=vanilla_anchor)
        # Load weights
        self.model.load_weights(weights_path, self.model.get_start_weight_dim())
        self.model.to(self.device, non_blocking=True)


    def set_Params(self, weights_path, model_cfg, th1, th2, *args, **kwargs):
        # self.th1 = th1
        # self.th2 = th2
        pass
        

    def detect(self, target_img):
        """
        Using the model for detecting cones in the target image.

        Args:
            target_img (np.array): the image used for cones detection.

        Returns:
            BB_list (list of BoundingBoxCone objects): list of the detected cones in the target image.
        """
        preprocessing_time = timer()
        new_width, new_height = self.model.img_size()
        pad_h, pad_w, ratio = calculate_padding(target_img.height, target_img.width, new_height, new_width)
        img = torchvision.transforms.functional.pad(target_img, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
        img = torchvision.transforms.functional.resize(img, (new_height, new_width))
        img = torchvision.transforms.functional.to_tensor(img)
        img = img.unsqueeze(0)
        preprocessing_time = timer() - preprocessing_time
        
        with torch.no_grad():
            img_to_device_time = timer()
            self.model.eval()
            img = img.to(self.device, non_blocking=True)
            img_to_device_time = timer() - img_to_device_time
            # output,first_layer,second_layer,third_layer = model(img)

            model_detection_time = timer()
            output = self.model(img)
            model_detection_time = timer() - model_detection_time

            post_processing_time = timer()
            # Tresholding detections
            for detections in output:
                detections = detections[detections[:, 4] > self.conf_thres]
                box_corner = torch.zeros((detections.shape[0], 4), device=detections.device)
                xy = detections[:, 0:2]
                wh = detections[:, 2:4] / 2
                box_corner[:, 0:2] = xy - wh
                box_corner[:, 2:4] = xy + wh
                probabilities = detections[:, 4]
                nms_indices = nms(box_corner, probabilities, self.nms_thres)
                main_box_corner = box_corner[nms_indices]
                probabilities_nms = probabilities[nms_indices]
                if nms_indices.shape[0] == 0:  
                    continue
            post_processing_time = timer() - post_processing_time
            
            bb_list_creation_time = timer()
            BB_list = []  # BB_list = [BB_1,BB_2,...,BB_N]
            # Extracting bounding boxes 
            for i in range(len(main_box_corner)):
                x0 = main_box_corner[i, 0].to('cpu').item() / ratio - pad_w
                y0 = main_box_corner[i, 1].to('cpu').item() / ratio - pad_h
                x1 = main_box_corner[i, 2].to('cpu').item() / ratio - pad_w
                y1 = main_box_corner[i, 3].to('cpu').item() / ratio - pad_h 
                pr = probabilities_nms[i]
                BB = BoundingBoxCone(u=round(x0), v=round(y0), h=round(y1 - y0), w=round(x1 - x0), pr=pr)
                # u, v - left top bounding box position in image plain
                # w, h - width and height of bounding box in pixels
                BB.cut_cone_from_img(target_img)
                color_prediction_time = timer()
                BB.predict_color() # classify detected cone to types
                color_prediction_time = timer() - color_prediction_time
                # cone_depth = predict_cone_depth(img_depth,BB)
                # BB = ['u','v','h','w','pr,'type'] ==>  (,'depth']) in the future we will have a deapth image
                BB_list.append(BB)
            bb_list_creation_time = timer() - bb_list_creation_time
            

            logging.debug("Preprocessing took: %d [ms]", 1000 *preprocessing_time)
            logging.debug("Image to device took: %d [ms]", 1000 *img_to_device_time)
            logging.debug("Running model took: %d [ms]", 1000 *model_detection_time)
            logging.debug("Postprocessing took: %d [ms]", 1000 *post_processing_time)
            logging.debug("Creating bounding box list took: %d [ms]", 1000 *bb_list_creation_time)
            logging.debug("Predicting 1 cone color took: %d [ms]", 1000 *color_prediction_time)

        return BB_list 


    def get_Params(self):
        # return self.th1, self.th2,
        pass

    def train(self):
        pass