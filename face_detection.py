import cv2
import mediapipe as mp
import time

mpfaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
FaceDetection=mpfaceDetection.FaceDetection()

cap=cv2.VideoCapture(0)
pTime = 0

while True:
    success,img=cap.read()
    imgrgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=FaceDetection.process(imgrgb)
    #print(results.detections)
    if results.detections:
        for id,detection in enumerate(results.detections):
            #mpDraw.draw_detection(img,detection)
            #print(id, detection)
            #print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic=img.shape
            bbox=int(bboxC.xmin*iw),int(bboxC.ymin*ih), \
                int(bboxC.width*iw),int(bboxC.height*ih)

            cv2.rectangle(img, bbox, (0,0,255),3)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, f'FPS:{str(int(fps))}', (30,40), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),3)

    cv2.imshow("img",img)
    cv2.waitKey(20)


if __name__=="__main__":
    main()