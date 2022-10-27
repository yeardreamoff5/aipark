import cv2
import numpy as np

class Distort:
    def __init__(self,image_path,save_image_name):
        self.image_path = image_path
        self.save_image_name = save_image_name

        # save as square image
        img = cv2.imread(image_path, 1)  # cv2.IMREAD_COLOR
        h,w,c = img.shape
        save_path = "data/raw_" + save_image_name + ".jpeg"
        if h != w:
            remove = int(abs(h - w) / 2)
            if h > w:
                self.crop_image = img[remove:h-remove,0:w]
            else:
                self.crop_image = img[0:h,remove:w-remove]
        else:
            self.crop_image = img.copy()
        cv2.imwrite(save_path, self.crop_image)

    def blur(self, k):# preprocess to blur
        blur_img = cv2.GaussianBlur(self.crop_image, (k, k), 0)
        save_path = "data/blur_" + str(k) + "_" + self.save_image_name + ".jpeg"
        cv2.imwrite(save_path, blur_img)

    def rotate(self, k):# preprocess to rotate
        # 이미지의 크기를 잡고 이미지의 중심을 계산합니다.
        (h, w) = self.crop_image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # 이미지의 중심을 중심으로 이미지를 45도 회전합니다.
        M = cv2.getRotationMatrix2D((cX, cY), k, 1.0)
        rotated = cv2.warpAffine(self.crop_image, M, (w, h))
        save_path = "data/rotate_" + str(k) + "_" + self.save_image_name + ".jpeg"
        cv2.imwrite(save_path, rotated)

        # rotate_dict = {90: cv2.ROTATE_90_CLOCKWISE, 180:cv2.ROTATE_180, 270:cv2.ROTATE_90_COUNTERCLOCKWISE}
        # rotate_image = cv2.rotate(self.crop_image, rotate_dict[k])
        # save_path = "data/rotate_" + str(k) + "_" + self.save_image_name + ".jpeg"
        # cv2.imwrite(save_path, rotate_image)

    # def adjust_hsv(self, k):# preprocess to val
    #     # img = cv2.imread(self.image_path, cv2.COLOR_BGR2HSV)
    #     adjust = np.full(self.crop_image.shape, (0, 0, k), dtype=np.uint8)
    #     hsv_image = cv2.add(self.crop_image, adjust)
    #     hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    #     save_path = "data/hsv_" + str(k) + "_" + self.save_image_name + ".jpeg"
    #     cv2.imwrite(save_path, hsv_image)

    def adjust_bright(self, k):
        # img = cv2.imread(self.image_path, -1)
        adjust = np.full(self.crop_image.shape, (k, k, k), dtype=np.uint8)
        bright_image = cv2.subtract(self.crop_image, adjust)
        save_path = "data/bright_" + str(k) + "_" + self.save_image_name + ".jpeg"
        cv2.imwrite(save_path, bright_image)

distort = Distort("../data/high_quality.jpg", "Af")
distort.blur(27)
distort.rotate(45)
# distort.adjust_hsv(50)
distort.adjust_bright(150)