import cv2

def sift_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(image, kp, None)
    return kp_image,kp,des

    kp = sift.detect(gray, None)
    kp, des = sift.compute(gray, kp)