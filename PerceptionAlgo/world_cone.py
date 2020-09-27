
class WorldCone:
    '''
    World Cone class for each detected cone in a image, a cone in 3D - world plain.

    Vars:
        x (float) - x coordinate in ENU coordinate system.
        y (float) - y coordinate in ENU coordinate system.
        id (int) - cone index.
        pr (float [0,1]) - the confidence of the detection.
        color (string) - Blue, Yellow or Orange as perception message type.

    '''
    # intialize class static variable
    index = 0

    def __init__(self, x, y, color, pr):
        WorldCone.index += 1
        self.id = index
        self.x = x
        self.y = y
        self.color = color
        self.pr = pr
        

        