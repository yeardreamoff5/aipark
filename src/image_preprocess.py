import cv2
import numpy as np

###divide image
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


###composit image
def composit(fg_image_path,bg_image_path,fg_size,height,weight,save_final_path):
    img_fg = cv2.imread(fg_image_path, cv2.IMREAD_UNCHANGED)
    img_fg = cv2.resize(img_fg,(fg_size,fg_size))
    img_bg = cv2.imread(bg_image_path)

    # 알파 채널을 이용해서 마스크와 역 마스크 생성
    _, mask = cv2.threshold(img_fg[:, :, 3], 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # 전경 영상 크기로 배경 영상에서 ROI 잘라내기
    img_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGRA2BGR)
    h, w = img_fg.shape[:2]
    roi = img_bg[height:height + h, weight:weight + w]

    # 마스크 이용해서 오려내기
    masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)  # OpenCV 이미지
    masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)  # 배경 이미지

    # 이미지 합성
    added = masked_fg + masked_bg
    img_bg[height:height + h, weight:weight + w] = added

    cv2.imwrite(save_final_path,img_bg)