import cv2
import numpy as np

'''
Class RectangleDetector.

'''
class RectangleDetector:

    # class contructor
    def __init__(self):
        
        self.img = None
        self.x_center, self.y_center, self.center_count = 0, 0 ,0
        self.contours = None

        self.MIN_GRAY = 240
        self.MAX_GRAY = 255

        self.minH, self.minS, self.minV = 30, 0, 177
        self.maxH, self.maxS, self.maxV = 150,92,255

        self.MIN_RATIO = 0.65
        self.MAX_RATIO = 1.15

        self.MIN_AREA = 250     # min observable good area = 400 (from 30m)
        self.MAX_AREA = 70000   # 55000

        self.MIN_POLYGON_SIDES = 4
        self.MAX_POLYGON_SIDES = 4

        self.show_only_biggest  = True
        self.shouldFilter       = True
        self.shouldSortByArea   = True

        self.isClosedContour    = True

        self.PERCENTAGE_PERIMETER =  0.02 # percentage of perimeter considered for polygon approximation

        self.old_x_center, self.old_y_center = 0,0
        self.alpha = 0.99 # weight to avoid center point moving too fast

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

    def drawContours(self, img, contours):
        pass

    def isWithinRanges(self, area, sides, ratio):
        return ((sides     >= self.MIN_POLYGON_SIDES   and sides    <= self.MAX_POLYGON_SIDES) 
            and (area      >= self.MIN_AREA            and area     <= self.MAX_AREA)
            and (ratio     >= self.MIN_RATIO           and ratio    <= self.MAX_RATIO))
    
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
        # print("area, sides, ratio = ", area, sides, ratio)
        return result

    def myContourArea(self, contour):
        area, sides, ratio, approx = self.getAreaSidesRatioApprox(contour)
        #print(area, sides, ratio)
        return cv2.contourArea(contour)

    def detectRectangles(self, imgToDraw=None, draw=False):

        if (self.contours == None):
            #print("returning")
            return

        if (imgToDraw is None):
            print("need an img")
            return

        count = 0
        stopFlag = False
        self.x_center, self.y_center, self.center_count = 0, 0, 0

        # filter out not-rectangles
        if (self.shouldFilter):
            self.contours = filter(self.isRectangle, self.contours)
        
        # if no contours satisfies our criteria, don't run
        if (len(self.contours) == 0):
            return

        # sort by area
        if (self.shouldSortByArea):
            self.contours = sorted(self.contours, key=lambda x: cv2.contourArea(x), reverse=True)
            
        # print("--------------------------------------------")
        # try:
        #     print("1st area  = ", cv2.contourArea(self.contours[0]))
        #     print("2nd area  = ", cv2.contourArea(self.contours[1]))
        #     print("last area = ", cv2.contourArea(self.contours[-1]))
        # except:
        #     pass

        while not stopFlag:
            # run until all contours have been found
            if (count > len(self.contours) - 1):
                stopFlag = True
                break

            # start by highest area
            contour = self.contours[count]
            count = count + 1
            
            area, sides, ratio, approx = self.getAreaSidesRatioApprox(contour)

            # since we already found the biggest area and it fits the criteria, we'll exit after this
            if self.show_only_biggest:
                stopFlag = True
                     
            # squeezing approx
            approx = np.squeeze(approx) 
            x_sum, y_sum = 0, 0 # initializing x and y coordinates that will be used to calculate center point
            
            if (len(approx) < 4):
                continue

            for corners in approx:
                x_sum += corners[0]
                y_sum += corners[1]
            
            # calculating center point
            x_avg = int(x_sum/sides)
            y_avg = int(y_sum/sides)
            
            self.x_center = self.x_center + x_avg
            self.y_center = self.y_center + y_avg
            self.center_count = self.center_count + 1

            if draw:
                # coordinate for writing text on img
                x1,y1 = contour[0][0]

                # putting text
                cv2.putText(imgToDraw, str(ratio)+'-'+str(sides)+'-'+str(area), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

                # drawing contour
                #imgToDraw = cv2.drawContours(imgToDraw, [contour], -1, (0,255,0), 2)                    
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                imgToDraw = cv2.drawContours(imgToDraw,[box],0,(0,255,0),2)

                cv2.circle(imgToDraw, (self.x_center, self.y_center), radius=1, color=(0,0,255), thickness=3)
 
    def findCenterPoint(self):

        # calculating center
        if (self.center_count != 0):
            self.x_center = int(self.x_center/self.center_count)
            self.y_center = int(self.y_center/self.center_count)
            cv2.circle(self.img, (self.x_center, self.y_center), radius=1, color=(0,0,255), thickness=3)
        else:
            self.x_center = None
            self.y_center = None

    def visionPipeline(self, visionArgs):
    
        img         = visionArgs['img']
        erode       = visionArgs['erode']
        showCanny   = visionArgs['showCanny']

        # Gaussian
        img = cv2.GaussianBlur(img, (5,5), 1)
        
        # HSV mask
        hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_mask= cv2.inRange(hsv, (self.minH, self.minS, self.minV),(self.maxH, self.maxS, self.maxV))
        thresh = hsv_mask

        # Erode
        if erode:
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.erode(thresh, kernel)


        threshold_canny = visionArgs['threshold_canny']
        apertureSize    = visionArgs['apertureSize_canny']

        # Canny
        canny = cv2.Canny(thresh, threshold_canny, threshold_canny * 2 ,)
        canny = cv2.dilate(canny,None, iterations=3)
        canny = cv2.erode(canny,None, iterations=1)

        self.contours, _ = cv2.findContours(canny, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

        if showCanny:
            cv2.imshow("canny", canny)

        return
        
    # main function
    def findRectanglesCenterPoint(self, img, visionArgs):
        
        self.visionPipeline(visionArgs)
        self.detectRectangles(imgToDraw=img, draw=True)
        # self.findCenterPoint()

        return img, self.x_center, self.y_center
