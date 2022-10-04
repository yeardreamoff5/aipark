import cv2
import numpy as np

def blur(image_path,save_image_name,k):

    img = cv2.imread(image_path,-1)
    blur_img = cv2.GaussianBlur(img,(k,k),0)
    save_path = "data/blur_" + str(k) + "_" + save_image_name +".jpeg"
    cv2.imwrite(save_path,blur_img)

def adjust_hsv(image_path,save_image_name,k):
    img = cv2.imread(image_path, cv2.COLOR_BGR2HSV)
    adjust = np.full(img.shape, (0, 0, k), dtype=np.uint8)
    hsv_image = cv2.add(img, adjust)
    save_path = "data/hsv_" + str(k) + "_" + save_image_name +".jpeg"
    cv2.imwrite(save_path,hsv_image)
blur("data/high_Lincoln.jpeg",5)
blur("data/high_Lincoln.jpeg",10)
adjust_hsv("data/high_Lincoln.jpeg","Lincoln",200)