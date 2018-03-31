# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import sys
import time
import cv2
x=0
y=0

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
font=cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
camera.vflip = True
camera.video_stabilization = True
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	out.write(image)
	
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(50, 50),
		#when the values are smallers, the face to detect can be smaller
		#flags=cv2.cv.CV_HAAR_SCALE_IMAGE

    )
    # DRAW A RECTANGLE AROUND THE FACES FOUND
	for (x, y, w, h) in faces:
        # ---To draw a rectangle this are the parameters
        # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
        # img is the image variable, it can be "frame" like in this example
        # x1,y1 ---------
        # |              |
        # |              |
        # |              |
        # -------------x2,y2
        # (255,0,0) are (R,G,B)
        # the last 2 is the thickness of the line 1 to 3 thin to gross
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 100), 1)

        #---To write the x,y on the middle of the rectangle.
		stringxy="+%s,%s"%(x,y) # To prepare the string with the xy values to be used with the cv2.putText function
        #In the case we want to put Xxvalue,Yyvalue we can use the following line removing #.
        #stringaxy="X%s,Y%s"%(x,y) 
		#cv2.putText(image,stringxy,(x+w/2,y+h/2),font, 1,(0,0,255),1)


    
	# DISPLAY THE RESULTING FRAME
	out.write(image)
	cv2.imshow("Frame", image)
	#print (x,y,)
	#print("\n")
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

out.release()
cv2.destroyAllWindows()