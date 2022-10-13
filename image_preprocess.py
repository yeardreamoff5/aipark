import cv2
import numpy as np
#
def divide_image(image_path,save_image_name):
    img = cv2.imread(image_path)
    h,w,c=img.shape
    # cv2.imshow("test",img)
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
    # cv2.waitKey(0)

    low_img = img[:,:int(w/2),:]
    high_img = img[:,int(w/2):,:]
    low_img_name = "data/low_"+ save_image_name + ".jpeg"
    high_img_name = "data/high_"+ save_image_name + ".jpeg"
    # cv2.imshow("low", low_img)
    cv2.imwrite(low_img_name,low_img)
    # cv2.imshow("high", high_img)
    cv2.imwrite(high_img_name,high_img)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()
    # cv2.waitKey(0)

divide_image("Lincoln.png","Lincoln")