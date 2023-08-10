import numpy as np
import cv2

class Motion_Detector:
    """
    Algorithm: 
        diff(current_frame,prev_frame) -> threshold -> GaussianBlur -> threshold -> dilate -> findContour -> return bounding box

    Description:
    ------------
        By computing the difference between image, the change hinted there is movement. Since our problem is a static camera,
        this is a fast and reliable method for moving object. The GaussianBlur and threshold makes it more robust towards random
        noise of the input. A min_contour_area was also use to prevent noise expanded by dilation. Use compute after init for
        detection.

    Methods:
    --------
     __init__(self,threshold_image_difference,min_countour_area):
        threshold_image_difference: minimum pixel difference to be consider as motion for pixel.
        min_countour_area: minimum area size to be consider object

    """
    def __init__(self,threshold_image_difference=20,min_countour_area=3000):
        """
        Initialize Motion_Detector

        Args:
            threshold_image_difference: minimum pixel difference to be consider as motion for pixel.
            min_countour_area: minimum area size to be consider object

        """
        self.frame_prev = None
        self.isDetected = False
        self._min_area = min_countour_area
        self._threshold = threshold_image_difference
        self._previous_bbs = []
    def compute(self,frame):
        """
        Compute difference between current_frame and previous_frame. If there is no previous frame, initialize and return []

        Algorithm: 
            diff(current_frame,prev_frame) -> threshold -> GaussianBlur -> threshold -> dilate -> findContour -> return bounding box

        Args:
            frame (nd.array): input_frame

        Returns:
            detected Objects (list): [[[x1,y1,w1,h1],0.1],[[x2,y2,w2,h2],0.1]....]
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not self.isDetected:
            self.isDetected = True
            self.frame_prev = frame_gray
            return []
        

        sub = np.abs(frame_gray.astype(np.float32) - self.frame_prev.astype(np.float32))
        sub = np.where(sub <= self._threshold, 0, 255).astype(np.uint8)
        sub = cv2.GaussianBlur(sub,(7,7),0)
        _,sub = cv2.threshold(sub,127,255,cv2.THRESH_BINARY)
        # for box,_ in self._previous_bbs:
        #     left,right,top,bottom = box[0], box[0] + box[2], box[1], box[1] + box[3]
        #     mask = np.zeros_like(sub)
        #     mask[top:bottom,left:right] = 255
        #     # About overlap region, fingers crossed. Hopefully, they are all dilated in the previous iteration. 
        #     kernel = np.ones((3, 3), np.uint8)
        #     temp = cv2.dilate(sub, kernel)
        #     temp = cv2.bitwise_and(temp, temp, mask=mask)
        #     sub = cv2.bitwise_or(sub,temp)
        sub = cv2.dilate(sub, np.ones((5, 5), np.uint8), iterations=5)
        cv2.imshow('Frame1',sub)
        contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bbs = []
        for contour in contours:
            # Calculate the moments to find the centroid
            area = cv2.contourArea(contour)
            if area>self._min_area:
                x, y, width, height = cv2.boundingRect(contour)
                # Assign a low confidence value
                bbs.append([[x, y, width, height],0.1])
        self.frame_prev = frame_gray
        self._previous_bbs = bbs
        return bbs
