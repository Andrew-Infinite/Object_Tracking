import cv2
import numpy as np
import time

from deep_sort_realtime.deepsort_tracker import DeepSort
from func.DNN_Detector import DNN_Detector
from func.Template_Detector import Template_Detector
from func.Motion_Detector import Motion_Detector
import func.util as util

template_detector = Template_Detector()
dnn_detector = DNN_Detector()
motion_detector = Motion_Detector()

cap = cv2.VideoCapture('../sample.mp4')
skip_counter = 400
count = 0


tracker = None
def algorithm(frame):
    """
    Algorithm:
        Detect Object -> Filter overlap box with NMS -> track object if tag is found, else template matching.

    Args:
        frame: Input frame

    Returns:
        Bounding box of Staff with [x,y,w,h], return [] if the staff was not yet found.

    Note:
        1. Object Detection were split into two parts:
            a. dnn_detector: yolov4-tiny
            b. motion_detector: difference between current_frame and previous_frame
        2. The dnn_detector had mode for invert because yolo_detector struggle to detect invert person
        3. Template Matching were done only done in region which was detected.

    """
    global tracker
    staff_box = []
    Tracking_BBS = []
    
    #moving obj
    Tracking_BBS = Tracking_BBS + motion_detector.compute(frame)
    
    #static obj
    Tracking_BBS = Tracking_BBS + dnn_detector.forward(frame,0.3)
    Tracking_BBS = Tracking_BBS + dnn_detector.forward(frame,0.3,is_invert=True)
    
    #remove overlap ROI
    Tracking_BBS = util.nms(Tracking_BBS,nms_threshold=0.2)
    
    try:
        #Tracking only start if there is a template match with 90% confidence on a person
        tracks = tracker.update_tracks(Tracking_BBS, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = int(track.track_id)
            left, top, right, bottom = track.to_ltrb()
            if track_id == 1:
                is_tracking = True
                staff_box = (int(left), int(top), int(right-left), int(bottom-top))
                # x,y,w,h = staff_box
                # print("staff:",x-w/2, y-h/2)
                # cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)),(0, 255, 0), 2)
                # label = f'{track_id:.2f}'
                # cv2.putText(frame, label, (int(left),  int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    except:
        # template_bbs = []
        for track in Tracking_BBS:
            img, coor = util.crop_img(frame,track[0])
            box = template_detector.template_matching(img)
            if box[1]>0.9:
                tracker = DeepSort(max_iou_distance=0.9,max_age=2)
                tracks = tracker.update_tracks(Tracking_BBS, frame=frame)
                # box[0][0] = box[0][0] + coor[0]
                # box[0][1] = box[0][1] + coor[1]
                # template_bbs.append(box)
                # tx,ty,tw,th = box[0]
                # x,y,w,h = track[0]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)
    
    return staff_box




while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        if(count<skip_counter):
            count = count+1
            continue
        
        start_time = time.time()
        box = algorithm(frame)
        if(len(box)==4):
            x,y,w,h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (x+w//2,y+h//2), 5, (0,255,0), -1)
            print("staff:",x+w/2, y+h/2)
        end_time = time.time()
        cv2.putText(frame, f'FPS: {1/(end_time - start_time):.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Frame',frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else: 
        break

cap.release()
cv2.destroyAllWindows()
