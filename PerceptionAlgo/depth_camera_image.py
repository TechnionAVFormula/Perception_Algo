import PIL
from .image import Image

class DepthImage(Image):
    '''
    Camera Deapth Image class for handeling depth image data and image operation.

    Vars:
        data - image pixel data from simulator in bit representation.
        width - image width.
        height - image height.
        type - uint16 / float32.
        hfov - horizontal camera field of view.
        vfov - vertical camera field of view.
        position - camera position in cognata car coordinate system.
        hsv - True or False, does the image is in hsv format.
    '''
    def __init__(self, data, w, h, typ, hfov, vfov, pos):
        """
        Constructor.

        Args:
            data ([type]): image pixel data from simulator in bit representation.
            w (float): image width.
            h (float): image height.
            typ (uint16 / float32): pixels type.
            hfov (float): horizontal camera field of view.
            vfov (float): vertical camera field of view.
            pos ([type]): camera position in cognata car coordinate system.
        """
        super().__init__(w, h)
        self.data = PIL.Image.frombytes("I;16", (w, h), data) # convert from bit representation to pixels
        self.position = pos
        self.hfov = hfov
        self.vfov = vfov
        self.type = typ