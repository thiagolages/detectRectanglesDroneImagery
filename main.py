from RectangleDetector import RectangleDetector
import cv2, os, sys
import numpy as np

# directory used for reading videos
video_dir = "videos"

# if you want the video to play by itself, set stepByStep to False
stepByStep = False
waitParam = 0 if stepByStep else 1

# whether to save the final result or not
saveResult      = False
resultFilename  = 'detectRectangles.avi'
saveFPS         = 30.0
saveResolution  = (640,480)

if (len(sys.argv) < 2 ):
    print("need video idx, from 0 to X")
    exit(-1)

# get all files
files = sorted(os.listdir(video_dir))
files = [f for f in files if ".txt" not in f] # remove any .txt files from the list

# get file ID passed in the cmd line
file_idx = int(sys.argv[1])

print("All available files = {}".format(files))
print("Working with video: {}".format(files[file_idx]))
print(("" if saveResult else "NOT ") + "Saving result..")

# get full filename
filename = os.path.join(video_dir, files[file_idx])

# open video
cap = cv2.VideoCapture(filename)

rect = RectangleDetector()

# if saveResult is enabled
if saveResult:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(resultFilename, fourcc, saveFPS, saveResolution)

while (cap.isOpened()):
    try:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            break
        
        posx, posy, channel = frame.shape
        posx += 250

        resultFrame          = frame.copy()
        visionArgs= {   'img'               : resultFrame,
                        'showCanny'         : False,
                        'erode'             : True,
                        'threshold_canny'   : 200,
                        'apertureSize_canny': 3,
                    }     

        resultFrame  , x, y  = rect.findRectanglesCenterPoint(resultFrame,visionArgs)

        cv2.imshow      ('img', frame)
        cv2.moveWindow  ("img", 0 , 0)

        cv2.imshow      ("resultFrame", resultFrame)
        cv2.moveWindow  ("resultFrame", posx ,  0  )
        
        if saveResult:
            out.write(resultFrame)

        if (cv2.waitKey(waitParam) == ord('q')):
            cv2.destroyAllWindows()
            cap.release()
            out.release()
            break

    except KeyboardInterrupt:
        print("Exiting..")
        cv2.destroyAllWindows()
        cap.release()
        exit()