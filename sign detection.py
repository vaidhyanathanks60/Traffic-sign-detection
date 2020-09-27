#opencv 2.4.13

import cv2
import numpy as np
import serial
import time
import subprocess as sub

port = serial.Serial("COM9", baudrate=9600, timeout=1)

##import urllib
MIN_MATCH_COUNT=40
lt=1;
gt=1;
detector=cv2.SIFT()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

trainImg=cv2.imread("weakbridge.jpg",0)
##cv2.imshow('train',trainImg)
trainImg2=cv2.imread("stop.jpg",0)
##trainImg3=cv2.imread("3.jpg",0)

trainKP1,trainDesc1=detector.detectAndCompute(trainImg,None)
trainKP2,trainDesc2=detector.detectAndCompute(trainImg2,None)
##trainKP3,trainDesc3=detector.detectAndCompute(trainImg3,None)

cam=cv2.VideoCapture(0)
##url="http://192.168.1.6:8080/shot.jpg"
while True:
##    imgPath=urllib.urlopen(url)
##    imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
##    QueryImgBGR=cv2.imdecode(imgNp,-1)
    ret, QueryImgBGR=cam.read()

    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    matches1=flann.knnMatch(queryDesc,trainDesc1,k=2)
    matches2=flann.knnMatch(queryDesc,trainDesc2,k=2)
##    matches3=flann.knnMatch(queryDesc,trainDesc3,k=2)

    
    for i in range(1,4):
        if i==1:
            matches=matches1;
            trainKP=trainKP1;
            trainDesc=trainDesc1;
        elif i==2:
            matches=matches2;
            trainKP=trainKP2;
            trainDesc=trainDesc2;
##        else:
##            matches=matches3;
##            trainKP=trainKP3;
##            trainDesc=trainDesc3;
        goodMatch=[]
        for m,n in matches:
            if(m.distance<0.75*n.distance):
                goodMatch.append(m)
        if(len(goodMatch)>MIN_MATCH_COUNT):
            tp=[]
            qp=[]
            for m in goodMatch:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp,qp=np.float32((tp,qp))
            H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
            h,w=trainImg.shape
            trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
            queryBorder=cv2.perspectiveTransform(trainBorder,H)
            #cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
##            print'#############  HELMET DETECTED  ############'
            if i==1:
                cv2.putText(QueryImgBGR, "WEAK BRIDGE!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print'#############  WEAK BRIDGE!  ############'
                port.write("## WEAK BRIDGE! ##")
                time.sleep(2)
            elif i==2:
                cv2.putText(QueryImgBGR, "STOP", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print'############# STOP ############'
                port.write("## STOP ##")
                time.sleep(2)
##            else:
##                cv2.putText(QueryImgBGR, "*********!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
##                print'#############  *********  ############'
        else:
            print "Scanning"
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()


