import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import shutil

# Function to extract SIFT features from an image
def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


save_path = "./human_image_extracted/template/"
image_paths = [save_path+file for file in os.listdir(save_path) if file.endswith(('.jpg'))]

# Extract SIFT features from all images and create a combined feature vector
sift_features = []
max_descriptors_length = 0
for path in image_paths:
    image = cv2.imread(path)
    _, descriptors = extract_sift_features(image)
    #print(descriptors.shape)
    if descriptors is not None:
        descriptors = descriptors.flatten()
        descriptors_length = descriptors.shape[0]
        if descriptors_length > max_descriptors_length:
            max_descriptors_length = descriptors_length
        sift_features.append(descriptors)
    else:
        sift_features.append(np.zeros(max_descriptors_length))
for i in range(len(sift_features)):
    descriptors_length = sift_features[i].shape[0]
    if descriptors_length < max_descriptors_length:
        sift_features[i] = np.concatenate([sift_features[i], np.zeros(max_descriptors_length - descriptors_length)])
        

sift_features = np.array(sift_features)


# Perform K-means clustering
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(sift_features)

# Display the cluster labels for each image
for i, path in enumerate(image_paths):
    shutil.move(path, save_path+str(cluster_labels[i]))
    #print(f"Image {i+1}: Cluster {cluster_labels[i]}")
