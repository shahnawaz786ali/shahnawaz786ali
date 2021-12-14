import cv2
import mediapipe as mp
import time 

class FaceDetector:
    def __init__(self,minDetectionCon=0.5):

        self.minDetectionConf=minDetectionCon
        self.mpfaceDetection=mp.solutions.face_detection
        self.mpDraw=mp.solutions.drawing_utils
        self.faceDetection=self.mpfaceDetection.FaceDetection()

    def faceDetector(self,img,draw=True):
        imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results=self.faceDetection.process(imgRGB)
        bboxs=[]
        if results.detections:
            for id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), \
                    int(bboxC.width*iw), int(bboxC.height*ih)

                bboxs.append([id, bbox, detection.score])
                if draw:
                    img=self.drawfancy(img,bbox)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

        return img,bboxs

    def drawfancy(self,img,bbox,l=15,t=5):
        x,y,w,h=bbox
        x1,y1=x+w,y+h

        cv2.rectangle(img, bbox, (0, 255, 0), 3)
        cv2.line(img, (x,y),(x+l, y), (0,0,255),t)
        cv2.line(img, (x, y), (x, y+l), (0, 0, 255), t)
        cv2.line(img, (x1,y),(x1-l, y), (0,0,255),t)
        cv2.line(img, (x1,y), (x1, y+l), (0, 0, 255), t)
        cv2.line(img, (x,y1),(x, y1-l), (0,0,255),t)
        cv2.line(img, (x, y1), (x+l, y1), (0, 0, 255), t)
        cv2.line(img, (x1, y1), (x1-l, y1), (0, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1-l), (0, 0, 255), t)

        return img
  
def main():
    cap=cv2.VideoCapture(0)
    pTime=0
    detector=FaceDetector()

    while True:
        success,img=cap.read()
        img,bboxs=detector.faceDetector(img)
        print(bboxs)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img, str(int(fps)), (30, 50),cv2.FONT_HERSHEY_PLAIN,3, (0, 255, 0), 3)
        cv2.imshow("img",img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()
