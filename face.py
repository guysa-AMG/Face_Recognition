import cv2 as cv
import socket as sk
import numpy as np
from PIL import Image
import uuid as uid
import time as t
import os

class Main:

    _MAX=65536
    _UDP_PORT=5050
    def __init__(self):
        self.sock :sk.socket= sk.socket(sk.AF_INET,sk.SOCK_DGRAM)
        self.sock.bind(("192.168.2.2",self._UDP_PORT))
        self.classifier=cv.CascadeClassifier("frontalface.xml")
        self.user = input("name:").strip(" ")

        if not os.path.isdir("./imgs/"+self.user):
            os.mkdir("./imgs/"+self.user)

    def recognition(self,frame):
        data =self.classifier.detectMultiScale(cv.cvtColor(frame,cv.COLOR_BGR2GRAY),12,1)
        faces =[]
        for x,y,w,h in data:
            print(f"x:{x},y:{y}----w:{w},h:{h}")
            cv.rectangle(frame,(x,y),(x+w,y+h),[0,255,0])
            name = uid.uuid3(uid.NAMESPACE_DNS,f"{(x,y)}")
            if frame is not None:
                cv.imwrite(f"./imgs/{self.user}/{self.user}-{name}.jpg",frame[y:h+y,x:w+x])
                t.sleep(1)


        return frame
    
    def modify(self,frame):
        frame= np.frombuffer(frame,np.uint8)
        frame = cv.imdecode(frame,cv.IMREAD_COLOR)
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame = cv.flip(frame,1)
        
        self.recognition(frame)
        return frame

    def view(self,frame):
        frame = self.modify(frame)

        if frame is not None:
            cv.imshow("Test param",frame)
            cv.waitKey(1)


    def handle(self):
        data,addr = self.sock.recvfrom(self._MAX)
        self.view(data)
       

    def run(self):
        while True:
            try:
                self.handle()
            except KeyboardInterrupt:
                print("\n...... closing up .........")
                break
        
        
        
if __name__ == "__main__":
    Main().run()