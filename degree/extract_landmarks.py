import dlib
import cv2
import math
import numpy as np
from imutils import face_utils
import glob
import pickle
import pandas as pd

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("degree/shape_predictor_68_face_landmarks.dat")
img_folder = ["1_QCIF","2_240p","3_360p","4_480p","5_720p","6_1080p","7_2K","8_4K"]

for folder_name in img_folder:
  print(folder_name)
  image_path = "data/" + folder_name
  img_list = sorted(glob.glob(image_path + '/' + '*.jpg'))
  # total_remarks = []
  for file in img_list:#default: img_list
    print(file)
    image = cv2.imread(file)
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    result = detector(img,1)
    for i, rect in enumerate(result):
      l = rect.left()
      t = rect.top()
      b = rect.bottom()
      r = rect.right()
      shape = predictor(img, rect)
      shape_np = face_utils.shape_to_np(shape).tolist()
    landmarks = [ shape_np[33],
                  shape_np[8],
                  shape_np[36],
                  shape_np[45],
                  shape_np[48],
                  shape_np[54] ]
    # print(shape_np)
    # print(landmarks)
    # total_remarks.append(landmarks)
    file_name = file.replace("data/","").replace(".jpg","")
    pickle_path = "data/" + file_name + ".txt"
    print(pickle_path)
    with open(pickle_path,"wb") as lf:
      pickle.dump(landmarks, lf)