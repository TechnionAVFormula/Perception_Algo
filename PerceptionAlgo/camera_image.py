import PIL
from .image import Image

class CameraImage(Image):
    """
    Camera Image class for handeling RGB image data and image operation.

    Vars:
        data - image RGB data format.
        width - image width.
        height - image height.
        hsv - True or False, does the image is in hsv format.
    """

    def __init__(self, data, w, h):
        """
        Constructor.

        Args:
            data ([type]): image pixel data from simulator in bit representation.
            w (float): image width.
            h (float): image height.
        """
        super().__init__(w, h)
        self.hsv = False
        self.data = PIL.Image.frombytes("RGB", (w, h), data, 'raw', 'RGBX', 0,-1) # convert from bit representation to RGB + depth format

    
    def draw_bb_on_image(self, bb_list):
        """
        Draw the detected boundong boxes on the image.

        Args:
            bb_list (list): list of BoundongBoxCone objects (after using cone.detector.detect()).

        Returns:
           img_with_bb (Image): the target image with the detected boundong boxes drawn on it. 
        """        
    
        img_with_bb = self.data
        draw = PIL.ImageDraw.Draw(img_with_bb)
        font = PIL.ImageFont.load_default()

        for cone in bb_list:
            # extract BB features:
            x0 = cone.u
            y0 = cone.v
            h = cone.h
            w = cone.w
            x1 = x0 + w
            y1 = y0 + h

            # draw BB + indicative text
            draw.rectangle((x0, y0, x1, y1), outline=cone.color)
            # text = f"({i}) {color}"
            # w_text, h_text = font.getsize(text)
            # draw.rectangle((x0, y0 - h_text, x0 + w_text, y0), fill=color)
            # draw.text((x0, y0-h_text),text , fill=(0, 0, 0, 128))

        return img_with_bb

    def convert_to_hsv(self):
        pass
        # self.hsv = True
    
    def is_hsv(self):
        return self.hsv
