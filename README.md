# Object_Tracking
This software is design for a interview problem, the video were taken from a Top-view wide angle camera with 25fps.
Task were to find and track a person with a special tag on it.

The program uses YOLOv4-tiny for person detection and diff(cur_frame,prev_frame) to detect object in motion, 
this was viable only because it was a static camera. 

The algorithm basically works like this:
    yolo detection + frame_diff detection -> nms filter -> template matching to find tag -> object tracking with DeepSORT

It was only able to run about 10fps during template matching, and 3.5fps during DeepSORT tracking.
