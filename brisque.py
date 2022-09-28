from brisque import BRISQUE
import cv2
import numpy as np
#
# img = cv2.imread("IQA.jpeg")
# h,w,c=img.shape
# cv2.imshow("test",img)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()
# cv2.waitKey(0)
#
# low_img = img[:,:int(w/2),:]
# high_img = img[:,int(w/2):,:]
#
# cv2.imshow("low", low_img)
# cv2.imwrite("low_img.jpeg",low_img)
# cv2.imshow("high", high_img)
# cv2.imwrite("high_img.jpeg",high_img)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()
# cv2.waitKey(0)

low_brisque, high_brisque = BRISQUE("low_img.jpeg", url=False), BRISQUE("high_img.jpeg", url=False)
print(f"low_image : {low_brisque.score()}, high_image : {high_brisque.score()}")