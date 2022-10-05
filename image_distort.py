import cv2
import numpy as np

class Distort:
    def __init__(self,image_path,save_image_name):
        self.image_path = image_path
        self.save_image_name = save_image_name

    def blur(self, k):
        img = cv2.imread(self.image_path, 1)
        # cv2.imwrite("data/AIPARK_raw_CEO.jpeg", img)
        blur_img = cv2.GaussianBlur(img, (k, k), 0)
        save_path = "data/blur_" + str(k) + "_" + self.save_image_name + ".jpeg"
        cv2.imwrite(save_path, blur_img)

    def adjust_hsv(self, k):
        img = cv2.imread(self.image_path, cv2.COLOR_BGR2HSV)
        adjust = np.full(img.shape, (0, 0, k), dtype=np.uint8)
        hsv_image = cv2.add(img, adjust)
        save_path = "data/hsv_" + str(k) + "_" + self.save_image_name + ".jpeg"
        cv2.imwrite(save_path, hsv_image)

    def adjust_bright(self, k):
        img = cv2.imread(self.image_path, 1)
        adjust = np.full(img.shape, (k, k, k), dtype=np.uint8)
        bright_image = cv2.subtract(img, adjust)
        save_path = "data/bright_" + str(k) + "_" + self.save_image_name + ".jpeg"
        cv2.imwrite(save_path, bright_image)

distort = Distort("data/AIPARK_raw_CEO.jpeg","AIPARK_CEO")
distort.adjust_bright(70)
#
# def blur(image_path,save_image_name,k):
#     img = cv2.imread(image_path,1)
#     # cv2.imwrite("data/AIPARK_raw_CEO.jpeg", img)
#     blur_img = cv2.GaussianBlur(img,(k,k),0)
#     save_path = "data/blur_" + str(k) + "_" + save_image_name +".jpeg"
#     cv2.imwrite(save_path,blur_img)
#
# def adjust_hsv(image_path,save_image_name,k):
#     img = cv2.imread(image_path, cv2.COLOR_BGR2HSV)
#     adjust = np.full(img.shape, (0, 0, k), dtype=np.uint8)
#     hsv_image = cv2.add(img, adjust)
#     save_path = "data/hsv_" + str(k) + "_" + save_image_name +".jpeg"
#     cv2.imwrite(save_path,hsv_image)
#
# def adjust_bright(image_path,save_image_name,k):
#     img = cv2.imread(image_path, 1)
#     adjust = np.full(img.shape, (k, k, k), dtype=np.uint8)
#     bright_image = cv2.subtract(img, adjust)
#     save_path = "data/bright_" + str(k) + "_" + save_image_name +".jpeg"
#     cv2.imwrite(save_path,bright_image)
#
# blur("data/AIPARK_raw_CEO.jpeg","AIPARK_CEO",7)
# blur("data/AIPARK_raw_CEO.jpeg","AIPARK_CEO",15)
# adjust_hsv("data/AIPARK_raw_CEO.jpeg","AIPARK_CEO",200)
# adjust_bright("data/AIPARK_raw_CEO.jpeg","AIPARK_CEO",100)