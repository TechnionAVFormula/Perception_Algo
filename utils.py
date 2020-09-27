
def compare_XYZ_to_GT (world_cones, world_cones_GT):
    """Get list of world cones detections and list of ground truth world cones and return index array of couples.
    connecting between the detected xyz and the closest ground truth xyz.

    Args:
        world_cones ([type]): [description].
        world_cones_GT ([type]): [description].

    Returns:
        indices (indices array): corresponding closest indices in GT array for each of the xyz detections.
    """    
    
    indices = [0]*len(world_cones)
    for i, cone in enumerate(world_cones):
        min_index, min_dist = 0, 0
        for i_GT, cone_GT in enumerate(xyz_cones_GT):
            dX = cone[0]-cone_GT[0]
            dY = cone[1] - cone_GT[1]
            dZ = cone[2] - cone_GT[2]
            dist = (dX**2+dY**2+dZ**2)**0.5
            if dist < min_dist or i_GT == 0:
                min_dist = dist
                min_index = i_GT
        indices[i] = min_index

    return indices
