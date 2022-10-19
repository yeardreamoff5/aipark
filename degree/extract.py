import dlib
import cv2
import math
import numpy as np
from imutils import face_utils
import glob
import pickle
import pandas as pd
from extract_degree import calculate_degree

img_folder = ["1_QCIF","2_240p","3_360p","4_480p","5_720p","6_1080p","7_2K","8_4K"]

class Extract:
  def __init__(self,img_folder,img_parent_path):
    self.img_folder = img_folder
    self.img_parent_path = img_parent_path

  def landmarks(self):#img_parent_path = "data/"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    for folder_name in self.img_folder:
      print(folder_name)
      image_path = self.img_parent_path + folder_name
      img_list = sorted(glob.glob(image_path + '/' + '*.jpg'))
      # total_remarks = []
      for file in img_list:  # default: img_list
        print(file)
        image = cv2.imread(file)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector(img, 1)
        for i, rect in enumerate(result):
          l = rect.left()
          t = rect.top()
          b = rect.bottom()
          r = rect.right()
          shape = predictor(img, rect)
          shape_np = face_utils.shape_to_np(shape).tolist()
        try:
          landmarks = [shape_np[33],
                       shape_np[8],
                       shape_np[36],
                       shape_np[45],
                       shape_np[48],
                       shape_np[54]]
          del shape_np
        except:
          landmarks = None
          print("null")
        # print(shape_np)
        # print(landmarks)
        # total_remarks.append(landmarks)
        file_name = file.replace(self.img_parent_path, "").replace(
          ".jpg", "")
        pickle_path = self.img_parent_path + file_name + ".txt"
        print(pickle_path)
        with open(pickle_path, "wb") as lf:
          pickle.dump(landmarks, lf)
        print(landmarks)

  def degree(self):
    for folder_name in self.img_folder:
      total_degree = []
      print(folder_name)
      image_path = self.img_parent_path + folder_name
      img_list = sorted(glob.glob(image_path + '/' + '*.jpg'))
      txt_list = sorted(glob.glob(image_path + '/' + '*.txt'))
      for img, txt in zip(img_list, txt_list):  # default: (img_list,txt_list)
        # print(img,txt)
        image = cv2.imread(img)
        with open(txt, 'rb') as lf:
          landmarks = pickle.load(lf)
        if landmarks == None:
          rotate_degree = [None, None, None]
        else:
          imgpts, modelpts, rotate_degree = calculate_degree(image, landmarks)
        print(landmarks)
        file_path = img.replace(self.img_parent_path, "")
        degree = [file_path, rotate_degree[0], rotate_degree[1], rotate_degree[2]]
        total_degree.append(degree)
        print(degree)
      pd_degree = pd.DataFrame(total_degree, columns=["img_path", "roll", "pitch", "yaw"])
      pd_degree.to_csv(self.img_parent_path + folder_name + ".csv")