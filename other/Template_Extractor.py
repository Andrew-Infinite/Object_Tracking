import cv2
import numpy as np

save_path = "./human_image_extracted/"
name_of_image = "human"
count_of_image = 1

# Load YOLO model and class names
net = cv2.dnn.readNet('yolov4-tiny.cfg', 'yolov4-tiny.weights')
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def algorithm(image):
    # Load input image
    height, width = image.shape[:2]
    # Preprocess input image
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)

    # Set the input to the model
    net.setInput(blob)
    # Forward pass through the network
    detections = net.forward(output_layers)
    
    Tracking_BBS = []
    # Process the detection results
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3 and class_id == 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)

                # make sure bbs always within range
                w = min(width,int(obj[2] * width))
                h = min(height,int(obj[3] * height))

                # make sure x,y not out of bound
                x = max(0, int(center_x - w / 2)); x = min(width-w, x)
                y = max(0, int(center_y - h / 2)); y = min(height-h, y)

                Tracking_BBS.append(([x,y,w,h],confidence,class_id))
    
    blob = cv2.dnn.blobFromImage(cv2.flip(image, 0), 0.00392, (416, 416), swapRB=True, crop=False)

    # Set the input to the model
    net.setInput(blob)
    # Forward pass through the network
    detections = net.forward(output_layers)
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3 and class_id == 0:
                center_x = int(obj[0] * width)
                center_y = int((1-obj[1]) * height)

                # make sure bbs always within range
                w = min(width,int(obj[2] * width))
                h = min(height,int(obj[3] * height))

                # make sure x,y not out of bound
                x = max(0, int(center_x - w / 2)); x = min(width-w, x)
                y = max(0, int(center_y - h / 2)); y = min(height-h, y)

                Tracking_BBS.append(([x,y,w,h],confidence,class_id))
                
    # Convert the bounding boxes to the format required by NMSBoxes (left, top, right, bottom)
    boxes = [box for (box, _,_) in Tracking_BBS]
    scores = [confidence for (_, confidence,_) in Tracking_BBS]
    # Define the IoU threshold for NMS (e.g., 0.5 means a 50% overlap is considered duplicate)
    nms_threshold = 0.3

    # Perform Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.2, nms_threshold)

    # Get the filtered bounding boxes after NMS
    Tracking_BBS = [Tracking_BBS[i] for i in indices]
    
    for bbs in Tracking_BBS:
        global count_of_image
        x,y,w,h = bbs[0]
        confidence = bbs[1]
        class_id = bbs[2]
        cv2.imwrite(save_path+name_of_image + str(count_of_image)+".jpg",image[y:y + h, x:x + w])
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #label = f'{classes[class_id]}: {confidence:.2f}'
        #cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        count_of_image = count_of_image + 1
    return image


cap = cv2.VideoCapture('../sample.mp4')

skip_counter=0
count = 0 
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        if(count<skip_counter):
            count = count+1
            continue
        # Display the resulting frame
        algorithm(frame)
        #cv2.imshow('Frame',algorithm(frame))

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
        # Break the loop
    else: 
        break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
