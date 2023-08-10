import cv2

def nms(Tracking_BBS,nms_threshold=0.3):
    """
    Non Maximum Suppression with opencv, the func do not filter by confidences. 
    Give your bounding box and we with remove overlap for you. :D

    Args:
        Tracking_BBS (list): [(x,y,w,h),confidence]
        nms_threshold (float): a number between 0.0 to 1.0

    Returns:
        Tracking_BBS (list): [(x,y,w,h),confidence]
    """
    boxes = [bbs[0] for bbs in Tracking_BBS]
    scores = [bbs[1] for bbs in Tracking_BBS]
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.0, nms_threshold)
    Tracking_BBS = [Tracking_BBS[i] for i in indices]
    return Tracking_BBS

def crop_img(img,box: tuple):
    """
    crop image and return left corner coordinate of the crop image (reference frame coordinate)

    Args:
        img (nd.array): input_image
        box (tuple): (x,y,w,h)

    Returns:
        crop_image, (x, y)
    """
    try:
        img = img[ box[1] : box[1] + box[3] ,
                     box[0] : box[0] + box[2] ]
        return img, (box[0], box[1])
    except:
        print("From Crop_img: empty_box")
        return img, (0,0)

class Bounding_Box:
    """
    Just a bounding box class.

    Description:
    ------------
    Support [x,y,w,h], [l,t,r,b], confidence query. Not use because convertion to DeepSORT format slow down a lot.
    init with __init__(self,box,confidence,is_ltrb=False), use is_ltrb if the input box is [l,t,r,b] instead of [x,y,w,h]

    Methods:
    --------
    x_top_left():
        return self._box[0]

    y_top_left():
        return self._box[1]

    x_bottom_right():
        return self._box[0] + self._box[2]

    y_bottom_right():
        return self._box[1] + self._box[3]

    w():
        return self._box[2]

    h():
        return self._box[3]

    box():
        return self._box

    confidence():
        return self._confidence

    ltrb():
        return (self.x_top_left(), self.y_top_left(), self.x_bottom_right(), self.y_bottom_right())

    """
    def __init__(self,box,confidence,is_ltrb=False):
        if is_ltrb:
            self._box = (box[0],box[1],box[2]-box[0],box[3]-box[1])
        else:
            self._box = box
        self._confidence = confidence

    def x_top_left(self):
        return self._box[0]
    
    def y_top_left(self):
        return self._box[1]
    
    def x_bottom_right(self):
        return self._box[0] + self._box[2]
    
    def y_bottom_right(self):
        return self._box[1] + self._box[3]
    
    def w(self):
        return self._box[2]
    
    def h(self):
        return self._box[3]
    
    def box(self):
        return self._box
    
    def confidence(self):
        return self._confidence
    
    def ltrb(self):
        return (self.x_top_left(), self.y_top_left(), self.x_bottom_right(), self.y_bottom_right())
    