import cv2, os, sys
import numpy as np


class HSVColoPicker:
    def __init__(self):        
        
        self.img_folder             = 'frames'
        self.img_idx                = 0 # to select img
        self.numImages              = None

        self.imgWindowName          = "img"
        self.hsvMaskWindowName      = "HSV Mask"
        self.hsvFilteredWindowName  = "HSV Filtered"
        self.threshWindowName       = "Thresh with erode"
        self.cannyWindowName        = "Canny"
        self.resultWindowName       = "Result"

        self.isSecondMonitor        = False
        self.xOffset                = 1920

        # self.imgWindowPos           = ( 0   + (self.xOffset if self.isSecondMonitor else 0),  0 )
        self.hsvFilteredWindowPos   = ( 0   + (self.xOffset if self.isSecondMonitor else 0),  0 )
        self.hsvMaskWindowPos       = (650  + (self.xOffset if self.isSecondMonitor else 0),  0 )
        self.threshWindowPos        = (1300 + (self.xOffset if self.isSecondMonitor else 0),  0 )
        self.cannyWindowPos         = ( 0   + (self.xOffset if self.isSecondMonitor else 0), 600)
        self.resultWindowPos        = (1300  + (self.xOffset if self.isSecondMonitor else 0), 600)
        
        self.filenames              = sorted(os.listdir(self.img_folder))
        self.numImages              = len(self.filenames)
        self.img_filename           = None

        self.canny_minVal           = 200
        self.canny_maxVal           = 200 * 2
        
        self.minH, self.minS, self.minV = 30, 0, 177
        self.maxH, self.maxS, self.maxV = 150,92,255

        self.x_center, self.y_center, self.center_count = 0, 0 ,0

        self.MIN_GRAY               = 0
        self.MAX_GRAY               = 255
        self.MIN_RATIO              = 0.65
        self.MAX_RATIO              = 1.15
        self.MIN_AREA               = 250     # min observable good area = 400 (from 30m)
        self.MAX_AREA               = 70000   # 55000
        self.MIN_POLYGON_SIDES      = 4
        self.MAX_POLYGON_SIDES      = 4
        self.show_only_biggest      = True
        self.shouldFilter           = True
        self.shouldSortByArea       = True
        self.isClosedContour        = True
        self.PERCENTAGE_PERIMETER   =  0.03 # percentage of perimeter considered for polygon approximation

        self.old_x_center, self.old_y_center = 0,0
        self.alpha = 0.99 # weight to avoid center point moving too fast

    def nextImage(self):
        self.img_idx += 1
        if (self.img_idx > self.numImages - 1):
            self.img_idx = 0
        
        self.start()

    def previousImage(self):
        self.img_idx -= 1
        if (self.img_idx < 0):
            self.img_idx = self.numImages - 1
        
        self.start()

    def isWithinRanges(self, area, sides, ratio):
        return ((sides     >= self.MIN_POLYGON_SIDES   and sides    <= self.MAX_POLYGON_SIDES) 
            and (area      >= self.MIN_AREA            and area     <= self.MAX_AREA)
            and (ratio     >= self.MIN_RATIO           and ratio    <= self.MAX_RATIO))
   
    def getContourRatio(self, contour):
    
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if (box.shape[0] > 4 or box.shape[1]>4):
            return
        p1, p2, p3, p4 = box
        distances = []
        distances.append(abs(np.linalg.norm(p1-p2)))
        distances.append(abs(np.linalg.norm(p2-p3)))
        w, h = min(distances),max(distances)
        
        return round(float(w)/h,3) # will always be <= 1.0

    def getAreaSidesRatioApprox(self, contour):
        
        # calculating contour area
        area = cv2.contourArea(contour)

        # approximating the countours to a polygon
        approx = cv2.approxPolyDP(contour, self.PERCENTAGE_PERIMETER*cv2.arcLength(contour, closed=self.isClosedContour), closed=self.isClosedContour)

        # calculating number of sides in polygon
        sides = len(approx)
        
        ratio = self.getContourRatio(contour) # always gets W and H in a way the ratio <= 1.0

        return area, sides, ratio, approx


    def isRectangle(self, contour):

        area, sides, ratio, approx = self.getAreaSidesRatioApprox(contour)

        # say if it's a rectangle or not based on area, sides and ratio AND if it's convex
        result = self.isWithinRanges(area, sides, ratio) #and cv2.isContourConvex(contour)
        
        return result

    def onMinGrayTrackbar(self, minGray):
        self.MIN_GRAY = minGray
        self.runPipeline()

    def onMaxGrayTrackbar(self, maxGray):
        self.MAX_GRAY = maxGray
        self.runPipeline()

    def onCannyMinValTrackbar(self, canny_minVal):
        self.canny_minVal = canny_minVal
        self.runPipeline()

    def onCannyMaxValTrackbar(self, canny_maxVal):
        self.canny_maxVal = canny_maxVal
        self.runPipeline()

    def onHminTrackbar(self, value):
        self.minH = value
        self.runPipeline()

    def onSminTrackbar(self, value):
        self.minS = value
        self.runPipeline()

    def onVminTrackbar(self, value):
        self.minV = value
        self.runPipeline()
    def onHmaxTrackbar(self, value):
        self.maxH = value
        self.runPipeline()

    def onSmaxTrackbar(self, value):
        self.maxS = value
        self.runPipeline()

    def onVmaxTrackbar(self, value):
        self.maxV = value
        self.runPipeline()

    def runPipeline(self):
        
        # gaussian
        self.img = cv2.GaussianBlur(self.originalImg, (5,5), 1)

        # HSV
        hsv     = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        hsv_mask= cv2.inRange(hsv, (self.minH, self.minS, self.minV),(self.maxH, self.maxS, self.maxV))

        hsv     = cv2.bitwise_and(hsv, hsv, mask=hsv_mask)

        # grayscale
        hsv = cv2.cvtColor(hsv , cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

        # using thresholding
        ret,thresh = cv2.threshold(gray,self.MIN_GRAY, self.MAX_GRAY,cv2.THRESH_BINARY)

        # erode
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.erode(thresh, kernel)
        #thresh = cv2.erode(thresh, kernel)
        #thresh = cv2.dilate(thresh, kernel)

        canny = cv2.Canny(thresh, self.canny_minVal, self.canny_maxVal)
        canny = cv2.dilate(canny,None, iterations=3)
        canny = cv2.erode(canny,None, iterations=1)
        #canny = cv2.Canny(self.img, self.canny_minVal, self.canny_maxVal)
        contours, _ = cv2.findContours(canny, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        # filter out not-rectangles
        if (self.shouldFilter):
            contours = filter(self.isRectangle, contours)

        # sort by area
        if (self.shouldSortByArea):
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        # if no contours satisfies our criteria, don't run
        if (len(contours) != 0):
            for contour in contours:
                #contour = contours[0]
                area, sides, ratio, approx = self.getAreaSidesRatioApprox(contour)

                approx = np.squeeze(approx) 
                x_sum, y_sum = 0, 0 # initializing x and y coordinates that will be used to calculate center point

                if (len(approx) < 4):
                    #print("len < 4")
                    continue
                    #exit()
                #print(approx)
                for corners in approx:
                    x_sum += corners[0]
                    y_sum += corners[1]

                # calculating center point
                x_avg = int(x_sum/sides)
                y_avg = int(y_sum/sides)

                x_center, y_center, center_count = 0, 0, 0

                x_center = x_center + x_avg
                y_center = y_center + y_avg
                center_count = center_count + 1
                
                # draw
                # coordinate for writing text on self.img
                x1,y1 = contour[0][0]

                # putting text
                cv2.putText(self.img, str(ratio)+'-'+str(sides)+'-'+str(area), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # drawing contour
                self.img = cv2.drawContours(self.img, [contour], -1, (0,255,0), 2)
                
                cv2.circle(self.img, (x_center, y_center), radius=1, color=(0,0,255), thickness=3)


                # ----- 
                # since we already found the biggest area and it fits the criteria, we'll exit after this
                if self.show_only_biggest:
                    break


        # cv2.imshow(self.imgWindowName           , self.originalImg  )
        cv2.imshow(self.hsvFilteredWindowName   , hsv               )
        cv2.imshow(self.hsvMaskWindowName       , hsv_mask          )
        cv2.imshow(self.threshWindowName        , thresh            )
        cv2.imshow(self.cannyWindowName         , canny             )
        cv2.imshow(self.resultWindowName        , self.img          )


    def loadImg(self):
        # get names
        self.img_filename        = self.filenames[self.img_idx]
        self.full_filename  = os.path.join(self.img_folder, self.img_filename)

        #load img
        self.originalImg    = cv2.imread(self.full_filename)
        self.originalImg    = cv2.resize(self.originalImg,(640,480))
        print("Loaded: {}".format(self.img_filename))

    def start(self):
        # load and run
        self.loadImg()
        self.runPipeline()

    def createTrackbars(self):
        # create trackbars
        cv2.createTrackbar( "minGray", self.threshWindowName    ,    self.MIN_GRAY  ,   255 , self.onMinGrayTrackbar    )
        cv2.createTrackbar( "maxGray", self.threshWindowName    ,    self.MAX_GRAY  ,   255 , self.onMaxGrayTrackbar    )
        cv2.createTrackbar( "minVal" , self.cannyWindowName     , self.canny_minVal ,   300 , self.onCannyMinValTrackbar)
        cv2.createTrackbar( "maxVal" , self.cannyWindowName     , self.canny_maxVal ,   600 , self.onCannyMaxValTrackbar)
        cv2.createTrackbar( "minH" ,    self.hsvMaskWindowName  ,     self.minH     ,   255 , self.onHminTrackbar       )
        cv2.createTrackbar( "maxH" ,    self.hsvMaskWindowName  ,     self.maxH     ,   255 , self.onHmaxTrackbar       )
        cv2.createTrackbar( "minS" ,    self.hsvMaskWindowName  ,     self.minS     ,   255 , self.onSminTrackbar       )
        cv2.createTrackbar( "maxS" ,    self.hsvMaskWindowName  ,     self.maxS     ,   255 , self.onSmaxTrackbar       )
        cv2.createTrackbar( "minV" ,    self.hsvMaskWindowName  ,     self.minV     ,   255 , self.onVminTrackbar       )         
        cv2.createTrackbar( "maxV" ,    self.hsvMaskWindowName  ,     self.maxV     ,   255 , self.onVmaxTrackbar       )
    
    def moveWindow(self, windowName, windowPos):
        print("moving {} to ({},{})".format(windowName, windowPos[0],windowPos[1]))
        cv2.moveWindow(windowName, windowPos[0], windowPos[1])
    
    def moveWindows(self):
        # self.moveWindow(self.imgWindowName        ,self.imgWindowPos          )
        self.moveWindow(self.hsvFilteredWindowName  ,self.hsvFilteredWindowPos  )
        self.moveWindow(self.hsvMaskWindowName      ,self.hsvMaskWindowPos      )
        self.moveWindow(self.threshWindowName       ,self.threshWindowPos       )
        self.moveWindow(self.cannyWindowName        ,self.cannyWindowPos        )
        self.moveWindow(self.resultWindowName       ,self.resultWindowPos       )


if __name__ == '__main__':

    picker = HSVColoPicker()
    picker.start()
    picker.createTrackbars()
    picker.moveWindows()

    print("Press D to go to next image, A to go to previous image.")
    print("Change any values in the HSV min and max ranges to suit your needs.")

    pressed = None
    while ( pressed != ord('q') ):
        pressed = cv2.waitKey(0)
        if (pressed == ord('d')):
            picker.nextImage()
        elif (pressed == ord('a')):
            picker.previousImage()
    
    cv2.destroyAllWindows()
    exit()