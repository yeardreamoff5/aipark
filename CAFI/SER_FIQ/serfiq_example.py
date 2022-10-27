# Customized by 25DREAM

# Author: Jan Niklas Kolf, 2020
from SER_FIQ.face_image_quality import SER_FIQ
import cv2
import os
import glob
import pandas as pd
import datetime 


def get_sefiq_score(img_folder_path,threshold):
    # Create the SER-FIQ Model
    # Choose the GPU, default is 0.
    ser_fiq = SER_FIQ(gpu=0)

    # Load the test image
    images = sorted(glob.glob(img_folder_path + '/' + '*.jpg'))
    a = 0
    total_list = []

    for image in images:
      a+=1
      test_img = cv2.imread(image)
      print(f"{a}/{len(images)} : SER-FIQ quality score of image {image}")

      # Align the image
      aligned_img = ser_fiq.apply_mtcnn(test_img)


      # Calculate the quality score of the image
      # T=100 (default) is sample good choice
      # Alpha and r parameters can be used to scale your
      # score distribution.
      try:
        score = ser_fiq.get_score(aligned_img, T=100)

        print(f"{a}/{len(images)} : SER-FIQ quality score of image {image} is", score)

      except:
        pass

      score_list = [image, score]
      total_list.append(score_list)

    # save the score list to csv
    # now = datetime.datetime.now()
    # file_date = now.strftime("%Y-%m-%d-%H-%M")
    csv_list = pd.DataFrame(data = total_list, columns=["img_path","serfiq_score"])
    csv_list = csv_list.sort_values(by="score", ascending=False)
    csv_list.to_csv(f"{img_folder_path}/serfiq.csv")
    serfiq_filter = csv_list[csv_list["serfiq_score"]>=threshold]
    return serfiq_filter