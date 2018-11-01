from scipy.spatial import distance as dist1
from imutils.video import VideoStream
from imutils import face_utils
# from threading import Thread
import numpy as np
# import playsound
import argparse
import imutils
import time
import dlib
import cv2


def face_movement(leftEye,rightEye,nose):
	A = dist1.euclidean(leftEye[0], nose[0])
	B = dist1.euclidean(rightEye[3], nose[0])

	return A,B

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist1.euclidean(eye[1], eye[5])
    B = dist1.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist1.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def mouth_aspect_ration(mouth):
    D1 = dist1.euclidean(mouth[13], mouth[19])   
    D2 = dist1.euclidean(mouth[14], mouth[18])   
    D3 = dist1.euclidean(mouth[15], mouth[17])   

    D4 = dist1.euclidean(mouth[12], mouth[16])


    mar = (D1 + D2 + D3) / (3.0 * D4)
    return mar




ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0,
    help="index of webcam on system")
args = vars(ap.parse_args())



MOUTH_AR_THRESH = 0.30
MOUTH_AR_CONSEC_FRAMES = 20
YAWN = 0
COUNTER = 0
COUNTER1 = 0



EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]


print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)


        

        ear = (leftEAR + rightEAR) / 2.0

        mouth = shape[mStart:mEnd]
        nose = shape[nStart:nEnd]
        mar = mouth_aspect_ration(mouth)

        lef_dist,rig_dist = face_movement(leftEye, rightEye, nose)

        nose = shape[nStart:nEnd]
        #
		
        mouthEyeHull = cv2.convexHull(mouth)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        

        cv2.drawContours(frame, [mouthEyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Eye Closing Detected!" , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0

        if mar > MOUTH_AR_THRESH:
            COUNTER1 += 1

            if COUNTER1 >= MOUTH_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Yawining Detected!" , (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
             
        else:
            COUNTER1 = 0

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Left_dist: {:.2f}".format(lef_dist), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "rig_dist: {:.2f}".format(rig_dist), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()