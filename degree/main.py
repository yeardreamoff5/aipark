from degree.extract_degree import face_orientation
import cv2
import math
import numpy as np
import glob
import pickle
import pandas as pd

img_folder = ["1_QCIF","2_240p","3_360p","4_480p","5_720p","6_1080p","7_2K","8_4K"]

for folder_name in img_folder:
  total_degree = []
  print(folder_name)
  image_path = "data/" + folder_name
  img_list = sorted(glob.glob(image_path + '/' + '*.jpg'))
  txt_list = sorted(glob.glob(image_path + '/' + '*.txt'))
  for img, txt in zip(img_list[:5], txt_list[:5]):#default: (img_list,txt_list)
    # print(img,txt)
    image = cv2.imread(img)
    with open(txt, 'rb') as lf:
      landmarks = pickle.load(lf)
      # print(landmarks)
    imgpts, modelpts, rotate_degree, nose = face_orientation(image, landmarks)
    file_path = img.replace("data/","")
    degree = [file_path,rotate_degree[0],rotate_degree[1],rotate_degree[2]]
    total_degree.append(degree)
    print(degree)
  pd_degree = pd.DataFrame(total_degree,columns=["img_path","roll","pitch","yaw"])
  pd_degree.to_csv("data/"+folder_name+".csv")