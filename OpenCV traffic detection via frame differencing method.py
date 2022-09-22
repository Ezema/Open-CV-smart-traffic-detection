# %% [markdown]
# # OpenCV road traffic detection

# %%
# Install openCV
!pip install opencv-python

# %%
import cv2

#Create a CV2 video capture object from filename'Traffic_Sample_1.mp4'
videoCaptureObject = cv2.VideoCapture('Traffic_Sample_1.mp4') 

# Exit if the video could not be opened
if not videoCaptureObject.isOpened(): 
    print("An error has occurred while trying to open the video")
    exit(1)

# Instantiate a KNN background substractor https://docs.opencv.org/3.4/de/de1/group__video__motion.html
backgroundSubstractorObjectInstance = cv2.createBackgroundSubtractorKNN(detectShadows = False)

# Initiate a loop that will iterate over every frame sequentially
while True:    
    
    # Read the video frame
    wasFrameReaded, videoFrame = videoCaptureObject.read()
    
    # If the frame had been successfully read continue, otherwise break the loop and stop the application
    if wasFrameReaded == True:        
    
        # For each of the frames of the video the first frame of the video (that is considered the background) will 
        # be substracted to obtain the difference between the two: the delta frame
        deltaFrame= backgroundSubstractorObjectInstance.apply(videoFrame)
        
        # Erosion: Removes pixels from the edges of objects in an image.
        # Dilation: Adds pixels to the edges of objects in an image.
        deltaFrame = cv2.erode(deltaFrame, None,20)
        deltaFrame = cv2.dilate(deltaFrame, None,10)               

        #The findContours function will retrieve the boundaries of the objects in the deltaFrame image
        contours,_=cv2.findContours(deltaFrame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
        # Create an reactangle overlay to show in the video the section that gets ignored 
        overlayRectangleX, overlayRectangleY, overlayRectangleWidth, overlayRectangleHeight = 3, 3, 1032, 259
        overlay = videoFrame.copy()
        cv2.rectangle(overlay, (overlayRectangleX, overlayRectangleY), (overlayRectangleX+overlayRectangleWidth, overlayRectangleY+overlayRectangleHeight), (0, 0, 0), -1)
        alpha = 0.6
        videoFrame = cv2.addWeighted(overlay, alpha, videoFrame, 1 - alpha, 0)
        
        # Create the 'Main Street' text and the rectangle that sourrounds it
        cv2.putText(videoFrame, 'Main Street', (5, 250), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255))
        cv2.line(videoFrame,(3,262), (1035,262), (0,0,255),3)
        cv2.line(videoFrame,(3,262), (3,595), (0,0,255),3)
        cv2.line(videoFrame,(1035,262), (1035,595), (0,0,255),3)
        cv2.line(videoFrame,(3,595), (1035,595), (0,0,255),3)
        
        # Iterate over the contours previously obtained.
        for contour in contours:
            # If the area of the contour is less than 3000 do not draw a bounding rectangle
            # The 3000 number was obtained from testing with with the video which number was the
            # minimum to avoid marking the people walking on the main street as well as the bicycles
            if cv2.contourArea(contour) < 3000:
                continue        
            # The boundingRect function calculates the tightest rectangle that encloses the contour object
            (x, y, width, height)=cv2.boundingRect(contour)
            # If the position of the countour's enclosing rectangle is outside the main street proceed to ignore it
            if y<262:
                continue
            # The countour's enclosing rectangle is inside the area of the main street
            cv2.rectangle(videoFrame, (x, y), (x+width, y+height), (0,255,0), 1)        
        
        # Name the playback window and set its dimensions. Consulted https://www.geeksforgeeks.org/python-opencv-resizewindow-function/
        cv2.namedWindow("Output video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Output video", 1920, 1080)        
        
        # While developing this notebook, showing the frameDifference mask is useful
        # Uncommenting the following code will show the mask window
        #cv2.namedWindow("deltaFrame Mask", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("deltaFrame Mask", 1920, 1080)            
        #cv2.imshow('deltaFrame Mask', deltaFrame)        
        
        # Show the output video that detects the moving cars
        cv2.imshow('Output video', videoFrame)
        
        # Press 'q' to 'quit'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # As mentioned above, this will get executed when the frame could not be read
        break
# When the video playback came to an end or the user pressed 'q', proceed to release the videoCapture object and close the window
videoCaptureObject.release()
cv2.destroyAllWindows()


