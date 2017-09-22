# Mirror the video camera

import cv2

# Main loop
def loop():
    # Input details
    width = 640
    height = 360
    fps = 60
    vc = cv2.VideoCapture(3)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    vc.set(cv2.CAP_PROP_FPS, fps)
    
    # Main GUI loop
    cv2.namedWindow("camera")
    frame = 0
    while True:
        ret, img = vc.read()
        
        key = cv2.waitKey(1)
        if key == 27:
            # Quit
            break

        frame += 1
        if img is not None:
            cv2.imshow("camera", img)

    vc.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    loop()
