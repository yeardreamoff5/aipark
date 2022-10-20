import dlib
import cv2
import math
import numpy as np
from imutils import face_utils
import glob
import pickle
import pandas as pd
from extract_degree import calculate_degree
import os

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
        out_face = np.zeros_like(image)

        result = detector(img, 1)
        for i, rect in enumerate(result):
          l = rect.left()
          t = rect.top()
          b = rect.bottom()
          r = rect.right()
          shape = predictor(img, rect)
          shape_mask = face_utils.shape_to_np(shape)
          shape_np = shape_mask.tolist()
          #initialize mask array
          remapped_shape = np.zeros_like(shape_mask) 
          feature_mask = np.zeros((image.shape[0], image.shape[1]))   
          # extract the face
          remapped_shape = cv2.convexHull(shape_mask)
          cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
          feature_mask = feature_mask.astype(np.bool)
          out_face[feature_mask] = image[feature_mask]
          
          
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
        mask_folder = self.img_parent_path + folder_name + "/result"
        try:
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)
        except OSError:
            print("Error: Failed to create the directory.")

        file_name_1 = file.replace(self.img_parent_path, "").replace(folder_name,"")
        mask_path = mask_folder + file_name_1
        
        cv2.imwrite(mask_path, out_face)
        print(mask_path)
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