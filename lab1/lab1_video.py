import numpy as np
import cv2

def recording():
    # Create a VideoCapture object to read from webcam and write to the file
    cam = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("output.avi",fourcc, 20.0, (640,480))
    # Read until video is completed
    while(cam.isOpened()):
        ret, frame = cam.read()
        if ret==True:
            # Display the frame
            cv2.imshow('frame',frame)
            # Writing frame to "output.avi"
            out.write(frame)
           # Press ESC to finish
            if cv2.waitKey(1) & 0xFF%256 == 27:
                break
        else:
            break
    # Release everything
    cam.release()
    out.release()


def processing():
    # Create a VideoCapture object to read from the file and write to the file
    cap = cv2.VideoCapture("output.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("output_final.avi",fourcc, 20.0, (640,480))
    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # Make frame grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Decompose gray to 3-dimension colors
            colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # Draw the blue line
            img_1 = cv2.line(colored,(0,0),(511,511),(255,20,0),5)
            # Draw the pink rectangle
            img_2 = cv2.rectangle(img_1,(122,122),(34,34),(133,0,255),5)
            # Display final frame
            cv2.imshow('Final Video', img_2)
            # Write final video
            out.write(img_2)
            # Press ESC to finish
            if cv2.waitKey(30) & 0xFF%256 == 27:
                break
        else:
            break
    cap.release()
    out.release()

recording()
processing()
cv2.destroyAllWindows()
