import numpy as np
from PIL import Image, ImageDraw
import cv2

class BoundingBoxCone:
    """
    Cone bounding box class for each detected cone in a image, a cone in 2D - image plain.

    Vars:
        u, v (float) - top left bounding box position in image plain.
        w, h (float) - width and height of bounding box in pixels.
        pr (float [0,1]) - the confidence of the detection.
        color (string) - "Blue", "Yellow" or "Orange"
        cone_image (np.array) - the bouding box image of the cone.

    """

    def __init__(self, u, v, h, w, pr):
        self.u = u
        self.v = v
        self.h = h
        self.w = w
        self.pr = pr
        self.color = None
        self.cone_image = None
    

    def cut_cone_from_image(self, target_img):
        """
        Cut the corresponding cone image from the full image.

        Args:
            target_img (np.array): the target image who have been used for the cone detection.
        """
        self.cone_image = target_img.data.crop((self.u, self.v,self.u + self.w, self.v + self.h))
    

    def predict_color(self):
                                              ############## Need to add orange ##################
        """
        Determine the cone color: Blue, Yellow or Orange.
        """
        frame = np.array(self.cone_image) # convert from PIL to cv
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #work on yelloew and blue cones
        img = cv2.cvtColor(hsv_frame, cv2.COLOR_BGR2HSV)

        #color range
        blueLow = np.array([90, 50, 50])
        blueHigh = np.array([150, 255, 255])
        yellowLow = np.array([20, 100, 100])
        yellowHigh = np.array([40, 255, 255])

        #different masks for image
        maskBlue = cv2.inRange(img, blueLow, blueHigh)
        maskBlue2 = cv2.dilate(maskBlue, np.ones((5, 5), np.uint8))
        maskBlue3 = cv2.erode(maskBlue2, np.ones((5, 5), np.uint8))
        maskYellow = cv2.inRange(img, yellowLow, yellowHigh)
        maskYellow2 = cv2.dilate(maskYellow, np.ones((5, 5), np.uint8))
        maskYellow3 = cv2.erode(maskYellow2, np.ones((5, 5), np.uint8))
        outputBlue = cv2.bitwise_and(img, img, mask=maskBlue3) ## 
        outputYellow = cv2.bitwise_and(img, img, mask=maskYellow3) ## 

        tempBlue = cv2.cvtColor(outputBlue, cv2.COLOR_BGR2GRAY)
        tempYellow = cv2.cvtColor(outputYellow, cv2.COLOR_BGR2GRAY)
        edgedBlue = cv2.Canny(tempBlue, 30, 300)
        edgedYellow = cv2.Canny(tempYellow, 30, 300)
        

        # final_frame = cv2.hconcat((frame, edgedYellow, edgedBlue))
        # cv2.imshow("final_frame", final_frame)

        n_white_pix_yellow = np.sum(maskYellow3 == 255)
        n_white_pix_blue = np.sum(maskBlue3 == 255)
        n_white_pix = [n_white_pix_yellow, n_white_pix_blue]
        max_value = max(n_white_pix)
        max_idx = n_white_pix.index(max_value)

        if max_idx == 0: # cone is yellow
            self.color = "yellow"
        elif max_idx == 1: # cone is blue
            self.color = "blue"
        # else: # cone is orange
        #     self.color = "orange"
            

    def get_mid_bb(self):
        """
        Extract the middle pixels of the bounding box. 

        Returns:
            (int): (u,v) the middle pixel of the boundong box.
        """
        return int(self.u + self.w/2), int(self.v + self.h/2)

        