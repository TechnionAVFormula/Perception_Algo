import cv2
import numpy as np
from PIL import Image
from .world_cone import WorldCone


# ---------------------------------------------------------------------------- #
#                                   Detection                                  #
# ---------------------------------------------------------------------------- #

# def trasform_img_cones_to_xyz(img_cones,width, height, depth_type, img_depth, h_fov, v_fov, camera_pos):
def img_cones_to_world_cones(img_cones, depth_img):
    """Get multiple BB in image plain (img_cones) and transform it to xyz coordinates of cognata car (xyz_cones).

    Args:
        img_cones (list): list of BoundingBoxCone objects.
        depth_img (CamerDepthImage):  depth image.

    Returns:
        cone_map (list): list of WorldCone objects (X,Y,Z,type) in cognata car coordinate system (X -forward, Y-left, Z-upward).
    """    

    # Extract parameters #### with the real camera need to prefor, only once with get_camera_params()!!!!
    K, R_inv, t_inv, cx, cy, f = extract_camera_params(depth_img.width, depth_img.height, depth_img.h_fov, depth_img.position)

    # Translating depth image from bytes to pixel array
    ###### depth_arr = np.asarray(Image.frombytes("I;16", (width, height), img_depth))
    depth_arr = np.asarray(depth_img.data)
    # Choose single representative point in each BB:
    img_cones_tmp = []
    for bb_cone in img_cones:
        img_cones_tmp.append(bb_cone.get_mid_bb()) # Get the u,v of center of BB, and type
    img_cones_tmp = np.asarray(img_cones_tmp)

    # Preparing indices and appropriate depths values
    index_x = img_cones_tmp[:, 0].astype(np.int)
    index_y = img_cones_tmp[:, 1].astype(np.int)
    depths = depth_arr[index_y, index_x]  # Using Matrix mult instead of loop to increase performance
    depths = depths/100  # convert from [cm] to [m]
    depths = convert_radial_to_perpendicular_depth(img_cones_tmp, depths, cx, cy, f)

    # Extract xyz coordinates of each cone together using matrices
    positions = world_XYZ_from_uvd(img_cones_tmp, depths=depths, K=K, R_inv=R_inv, t_inv=t_inv)

    # Arrange the data 
    cone_map = []  # list of WorldCone objects in ENU coordinate system (X - right, Y-forward, Z-upward)
    for index, bb_cone in enumerate(img_cones):
        # img_depth_px = img_depth.load()
        cone = WorldCone(*positions[index,0:2], bb_cone.color, bb_cone.pr)
        cone_map.append(cone)

    return cone_map



def extract_camera_params(width, height, h_fov, camera_pos):
    """Extract Camera parameters from he image.

    Args:
        width (float): image width.
        height ([type]): image height.
        h_fov (deg): horizontal fieald of view.
        camera_pos (meters or cm): camera relative position on vehicle.

    Returns:
        k: camera matrix.
        R_inv: camera rotaion vector (inverse).
        t_inv: camera translation vector (inverse).
        cx, cy: camera optical center.
        f: camera focal length.
    """
    # Focal length of the camera
    h_fov = h_fov*np.pi / 180 # in [rad]
    f = width / (2 * np.tan(h_fov / 2)) # 1/2w        (hFOV)
                                        # ----  = tan (----)
                                        #  f          (  2 )
    # Camera pin hole position on image
    cx = width / 2
    cy = height / 2 - 3  # horizon is 3 pixels above the centerline

    # Camera matrix
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float)

    # Transformation from camera -> cognata car coordinate system
    angle = - np.pi / 2.0

    #  Transformation from default camera cord. system (z front, x right, y down) to ENU
    R_ENU = np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]], dtype=np.float)

    #  Transformation from ENU to Cognata (Car) cord system (z up,x front,y left)- get from message: camera rot              
    R_ENU2cognata = np.array([[0, 1, 0],
                              [-1, 0, 0],
                              [0, 0, 1]], dtype=np.float)

    # In the future we will get from Cognata the R matrix, need to check what we get.                           
    R_inv = R_ENU2cognata@R_ENU

    # Camera position in cognata car coordinate system
    t_inv = np.array([camera_pos.x,camera_pos.y,camera_pos.z] , dtype=np.float)

    return K, R_inv, t_inv, cx, cy, f

# def trasform_img_point_to_xyz(img_point, img_depth, h_fov, v_fov, width, height):
#     # extract parameters
#     u = img_point[0]
#     v = img_point[1]
#     alpha_h = (180 - h_fov)/2  # [deg]
#     alpha_v = (180 - v_fov)/2  # [deg]
#     # calculating gammas:
#     gamma_h = alpha_h + (1-u / width) * h_fov  # [deg]
#     gamma_v = alpha_v + (v / height) * v_fov  # [deg]
#     # calculating X,Y,Z in ENU coordinate system (X - right, Y-forward, Z-upward)
#     Y = img_depth
#     X = img_depth / np.tan(gamma_h * np.pi / 180)
#     Z = img_depth / np.tan(gamma_v * np.pi / 180)

#     return [X, Y, Z]

# def inverse_perspective(R, t):
#     Ri = np.transpose(R)  # for a rotation matrix, inverse is the transpose
#     ti = -Ri @ t
#     return Ri, ti



def convert_radial_to_perpendicular_depth(points, depths, cx, cy, f):
    """Converts radial depth to perpendicular depth in image plain.

    Args:
        points (array): points in the image 2D plain.
        depths (array): radial depth values for each point in points.
        cx (float): image principal point horizontal pixel index.
        cy (float): image principal point vertial pixel index.
        f (float): camera focal length.

    Returns:
        (array): corresponding perpendicular depth values.
    """    

    return depths*f/(f**2 + (points[:, 0]-cx)**2 + (points[:, 1]-cy)**2)**0.5




def world_XYZ_from_uvd(points, depths, K, R_inv, t_inv):
    """Transforming uvd (uv in 2D image plain + depth) to xyz in world coordinates using matrices.

    Args:
        points (np.array): points in the image 2D plain.
        depths (np.array): radial depth values for each point in points.
        K (np.array): camera matrix.
        R_inv (np.array): transformation from camera coordinate system to world coordinate system.
        t_inv (np.array): camera position in world coordinates.

    Returns:
        positions (np.array): position of requested points in world coordinates.
    """    
    
    K_inv = np.linalg.inv(K)
    uv1 = cv2.convertPointsToHomogeneous(points)[:,0,:]
    # s(u,v,1) = K(R(xyz)+t)
    # xyz = Rinv*(Kinv*s*(uv1)) + tinv
    image_vectors = np.multiply(uv1.T, [depths]) # depths is broadcast over the 3 rows of uv.T
    positions = (R_inv@K_inv@image_vectors).T + t_inv  # tinv automatically broadcast to all matrix rows

    return positions

   
# ---------------------------------------------------------------------------- #
#                                  Calibration                                 #
# ---------------------------------------------------------------------------- #

def inverse_perspective(R, t):
    Ri = np.transpose(R)  # for a rotation matrix, inverse is the transpose
    ti = -Ri @ t
    return Ri, ti
    


def World_XY_from_uv_and_Z(imgpoints, K, R, t, Z):
    imgpoints_h = cv2.convertPointsToHomogeneous(imgpoints) # Turns the 2d cordinate (u,v) to a 3d (u,v,1)
    Rinv = inv(R)
    Kinv = inv(K)
    # objpoints = np.zeros((len(imgpoints), 3))  # Possible to give type >>> dtype=np.float64
    objpoints = []
    obj_vector = Rinv @ t

    for i in range (len(imgpoints)):
        uv = np.squeeze(imgpoints_h[i])
        uv = imgpoints_h[i].T
        img_vector = Rinv @ Kinv @ uv

        s = (Z + obj_vector[2])/(img_vector[2])
        obj_pos = Rinv @ (s * Kinv @ uv - t)

        objpoints.append(obj_pos)

    return objpoints