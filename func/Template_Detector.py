import cv2
import os

class Template_Detector:
    """
    A template matching based detector. Give me the template, I will tell you where it best fit.

    Description:
    ------------
    init with __init__(self,template_path,template_file_type), you can have more than one template in the directory,
    the program would look for all same filetype within the given path. use template_matching to find match

    Methods:
    --------
    template_matching(self,image,mode):
        use this to detect template.

    """
    def __init__(self,template_path = './template',template_file_type = 'jpg'):
        """
        Initialize Template_Detector

        Args:
            template_path: path/to/template
            template_file_type: .jpg or .png or any type cv2.imread support

        """
        template = [os.path.join(template_path, file) for file in os.listdir(template_path) if file.endswith((template_file_type))]
        self.templates = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in template]
    def template_matching(self,image,mode=0):
        """
        template matching, help your image to find it's partner.

        Args:
            image (nd.array): input_image
            mode (int):
                1) _max_confidence_compare_with_all_template:
                    brute force all template and find the location with best NCC score.
                2) _max_average_of_all_template:
                    brute force all template and stack the result NCC image, then find location with highest NCC score.
                    This struggle to get a good confident score except if your number of template is about the same as
                    the Area of the input.

        Returns:
            Best_fit_location (list): [[x,y,w,h],confidence]]
        """
        if mode == 0:
            box = self._max_confidence_compare_with_all_template(image)
        elif mode == 1:
            box = self._max_average_of_all_template(image)

        return box
    
    def _max_confidence_compare_with_all_template(self,img):
        """
        brute force all template and find the location with best NCC score.

        Args:
            image (nd.array): input_image

        Returns:
            Best_fit_location (list): [[x,y,w,h],confidence]]
        """
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        bbs = []
        highest_confidence = 0

        for template in self.templates:
            w, h = template.shape[1],template.shape[0]
            res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc

            if(max_val>highest_confidence):
                highest_confidence = max_val
                bbs = [[top_left[0],top_left[1],w,h],max_val]

        return bbs
        
    def _max_average_of_all_template(self,img,confidence_threshold=0.01):
        """
        brute force all template and stack the result NCC image, then find location with highest NCC score.
        This struggle to get a good confident score except if your number of template is about the same as
        the Area of the input.

        Args:
            image (nd.array): input_image
            confidence_threshold (float): a number between 0.0 to 0.1

        Returns:
            Best_fit_location (list): [[x,y,w,h],confidence]]
        """
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        bbs = []
        w,h = 0,0
        for template in self.templates:
            w, h = (template.shape[1],template.shape[0]) if (w<=template.shape[1] and h<=template.shape[0]) else (w,h)
            try:
                res = res + cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
            except:
                res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        max_val = max_val/len(self.templates)
        
        if max_val > confidence_threshold:
            bbs = [[top_left[0],top_left[1],w,h],max_val]
        else:
            bbs = ((0,0,0,0),0)
        return bbs