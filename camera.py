import numpy as np
import cv2
import glob
import pyzed.sl as sl
# from ReadData import read_alignment, read_calibration
from Geometry import  World_XY_from_uv_and_Z, inverse_perspective
# from utils import getBoxes


class Camera:

    def __init__(self):
        # Create a Camera object
        zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080  # Use HD1080 video mode, should be res
        init_params.camera_fps = 30  # Set fps at 30, should be fps

        # Open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        image_left = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()

        # Get the Camera Matrix and distortion coefficients
        Calib_data = load('Calibration.npz') 
        # Calib_data.files >>> ['CameraMtx', 'DistortionVec']
        K = Calib_data['CameraMtx'] 
        d = Calib_data['DistortionVec']

        self.zed = zed
        self.left_image = image_left  # self.right_image =
        self.params = runtime_parameters
        self.K = K
        self.d = d
        self.t = None
        self.R = None


    def set_Params(self, zed, fps, res):

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.res # Use HD1080 video mode, should be res
        init_params.camera_fps = fps  # Set fps at 30, should be fps

        # Open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        image_left = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()

        self.zed = zed
        self.left_image = image_left  # self.right_image =
        self. params = runtime_parameters



    def calibration():
        
        # termination criteria - of the form: (type, max_iter, epsilon)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (3,0,0), (6,0,0) ....,(21,0,12) - cordinate of the chessboard
        #need to specify the correct chessboard grid size - here it is 8x5 and square size is 30mm.
        objp = np.zeros((5*8,3), np.float32)
        objp[:,0:3:2] = (np.mgrid[0:8,0:5].T.reshape(-1,2))*3

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # Return a possibly-empty list of path names that match pathname(*.jpg)
        # Get the images from the folder
        images = glob.glob('50images\*.png')
        # For the 
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners for each image. return the corners points in image
            # and retval will be true if the pattern is obtained
            # Need to specify the correct chessboard pattern size of inner corners - here it is 8x5
            # Note: The function requires white space (like a square-thick border, the wider the better) around
            # the board to make the detection more robust in various environments. 
            ret, corners = cv2.findChessboardCorners(gray, (8,5),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp) 

                # increase the accuracy of the corners
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
                
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (8,5), corners2,ret)
                cv2.imshow('img',img)
                cv2.waitKey(5)

        cv2.destroyAllWindows()   
        # Returns the camera matrix, distortion coefficients, rotation and translation vectors.
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        np.savez('Calibration.npz', **{'CameraMtx': mtx, 'DistortionVec': dist})


    def alignment(self): # need to add flag for right or left or do both together

        # Grab an image, a RuntimeParameters object must be given to grab()
        if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            # Each new frame is added to the SVO file
            
            self.zed.retrieve_image(self.left_image, sl.VIEW.VIEW_LEFT)
            # Get the timestamp at the time the image was captured
            timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.TIME_REFERENCE_CURRENT)  
            print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(self.left_image.get_width(), self.left_image.get_height(),
                    timestamp))

            # To recover data from sl.Mat to use it with opencv, we use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            img = self.left_image.get_data()

            cv2.imshow("Image", img)
            cv2.waitKey(2500)


        # termination criteria - of the form: (type, max_iter, epsilon)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (3,0,0), (6,0,0) ....,(21,0,12) - cordinate of the chessboard
        #need to specify the correct chessboard grid size - here the inner cotners size sre 8x5 and square size is 3cm.
        objp = np.zeros((5*8,3), np.float32)
        objp[:,0:3:2] = (np.mgrid[0:8,0:5].T.reshape(-1,2))*3
        #objp += (0,200,0)

        # Arrays to store object points and image points from the image.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        ###img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners for each image. return the corners points in image
        # and retval will be true if the pattern is obtained
        # Need to specify the correct chessboard pattern size of inner corners - here it is 8x5
        # Note: The function requires white space (like a square-thick border, the wider the better) around
        # the board to make the detection more robust in various environments. 
        ret, corners = cv2.findChessboardCorners(gray, (8,5),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp) 

            # increase the accuracy of the corners
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (8,5), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(2500)

        # Returns the camera matrix, distortion coefficients, rotation and translation vectors.
        # rvec - Axis with angle magnitude (radians) [x, y, z]

        retval, r_inv, t_inv = cv2.solvePnP(objectPoints=objpoints[0], imagePoints=imgpoints[0], cameraMatrix=self.K, distCoeffs=self.d)

        #tvec.resize(3,1)
        # Transform the rotation vector to Rmat - Rotation Matrix (radians)
        R_inv, _ = cv2.Rodrigues(r_inv)

        # Saving the paramaters
        np.savez('Alignment.npz', **{'RotationMtx': R_inv, 'TranslationVec': t_inv})
        R, t = inverse_perspective(R_inv, t_inv)
        self.R = R
        self.t = t


    def take_img(self): # need to to flag left or right camera or take both
    
        # Grab an image, a RuntimeParameters object must be given to grab()
        if self.zed.grab(self.params) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            # Each new frame is added to the SVO file
            
            zed.retrieve_image(self.left_image, sl.VIEW.VIEW_LEFT)
            # Get the timestamp at the time the image was captured
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.TIME_REFERENCE_CURRENT)  
            print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(self.left_image.get_width(), self.left_image.get_height(),
                    timestamp))

            # To recover data from sl.Mat to use it with opencv, we use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            img = self.left_image.get_data()

            return img


    
    def take_Vid(self):
        pass


    def close(self):
        self.zed.close()


    def save_Image(img):
        pass
