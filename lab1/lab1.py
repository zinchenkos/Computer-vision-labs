import numpy as np
import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("capture")

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("capture", frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame.png"
        cv2.imwrite(img_name, frame)
        break

cv2.destroyAllWindows()

while True:
        img_ = cv2.imread("opencv_frame.png", cv2.IMREAD_ANYCOLOR)
        cv2.imshow("raw0",img_)
        key = cv2.waitKey(1)
        if key == ord('z'):
            # 'z' pressed
            print("Processing image...")
            print("Converting RGB image to grayscale...")
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            print("Converted RGB image to grayscale...")
            colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR);
            print("Drawing a blue line...")
            img_1 = cv2.line(colored,(0,0),(511,511),(255,20,0),5)
            print("Drawed a blue line...")
            print("Drawing a red rectangle...")
            img_2 = cv2.rectangle(img_1,(122,122),(34,34),(13,0,255),5)
            print("Drawed a red rectangle...")
            img_final_name = "opencv_frame_processed.png"
            cv2.imwrite(img_final_name, img_2)
            window_name = "processing_image"
            cv2.namedWindow(window_name)
            cv2.imshow(window_name, img_2)
            #key = cv2.waitKey(1)
        if key%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

cam.release()
cv2.destroyAllWindows()
