import cv2
import numpy as np

class DNN_Detector:
    """
    A dnn detector with opencv. Give me the model, I give you the result.

    Description:
    ------------
    init then use forward to detect object.

    Methods:
    --------
    __init__(self,cfg_path,weights_path):
        initialization
    forward(image,confidence_threshold,classes,is_invert):
        use this function to do object detection

    """
    def __init__(self,cfg_path = './model/yolov4-tiny.cfg',weights_path = './model/yolov4-tiny.weights'):
        """
        Initialize DNN_Detector

        Args:
            cfg_path: path/to/.cfg
            weights_path: path/to/.weights

        """
        self._net = cv2.dnn.readNet(cfg_path, weights_path)

        # Get output layer names
        layer_names = self._net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self._net.getUnconnectedOutLayers()]

    def forward(self,image,confidence_threshold,classes={0},is_invert=False):
        """
        Object Detection function, give me the image, I give you the result.

        Args:
            image (nd.array): input_image
            confidence_threshold (float): remove bounding box bellow threshold, value between 0.0 to 1.0
            classes (set): the class index for object to be detected. 0 for yolov4-tiny is for the person class
            is_invert (boolean): Invert the image before detection.

        Returns:
            detected Objects (list): [[[x1,y1,w1,h1],confidence1],[[x2,y2,w2,h2],confidence2]....]
        """
        if is_invert:
            image = cv2.flip(image.copy(), 0)

        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
        self._net.setInput(blob)
        detections = self._net.forward(self.output_layers)
        
        Tracking_BBS = []
        # Process the detection results
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold and class_id in classes:
                    center_x = int(obj[0] * width)
                    center_y = int((1-obj[1]) * height) if is_invert else int(obj[1] * height)

                    # make sure bbs always within range
                    w = min(width,int(obj[2] * width))
                    h = min(height,int(obj[3] * height))

                    # make sure x,y not out of bound
                    x = max(0, int(center_x - w / 2)); x = min(width-w, x)
                    y = max(0, int(center_y - h / 2)); y = min(height-h, y)

                    Tracking_BBS.append([[x,y,w,h],confidence])
        return Tracking_BBS