# Import OpenCV2 for image processing
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time

# Start capturing video 
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
camera.vflip = True
camera.video_stabilization = True
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)

#vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, one face id
face_id = 5

# Initialize sample face image
count = 0

# Start looping
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    # Capture video frame
	image = frame.array
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
	faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loops for each faces
	for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
		cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # Increment sample face image
		count += 1

        # Save the captured image into the datasets folder
		cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        # Display the video frame, with bounded rectangle on the person's face
		cv2.imshow('frame', image)

    # To stop taking video, press 'q' for at least 100ms
	rawCapture.truncate(0)
	if cv2.waitKey(100) & 0xFF == ord('q'):
		break

    # If image taken reach 100, stop taking video
	elif count>100:
		break

# Stop video
#vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()
