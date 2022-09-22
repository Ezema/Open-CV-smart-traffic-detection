# %% [markdown]
# # OpenCV road traffic detection plus region of interest counting

# %% [markdown]
# ## Introduction
# 
# The moving cars are successfully detected using the frame differencing and background subtraction techniques.
# To detect when those detected cars go to the city centre, a rectangle-shaped [Region of interest](https://en.wikipedia.org/wiki/Region_of_interest) was created (visually represented by coloring it with orange). When the cars pass through the region of interest, the counter gets updated.

# %%
# Install openCV
!pip install opencv-python

# %%
import cv2

# Create the function that will be used for both videos
def calculateCityCentreTrafficFromVideoPathAndShowVideo(videoPath):
    #Create a CV2 video capture object from videoPath
    videoCaptureObject = cv2.VideoCapture(videoPath) 

    # Exit if the video could not be opened
    if not videoCaptureObject.isOpened(): 
        print("An error has occurred while trying to open the video")
        exit(1)

    # Get the video playback length
    framesPerSeconds = videoCaptureObject.get(cv2.CAP_PROP_FPS) 
    totalNumberOfFrames = int(videoCaptureObject.get(cv2.CAP_PROP_FRAME_COUNT))
    videoPlaybackLengthInSeconds = totalNumberOfFrames/framesPerSeconds
        
    # Instantiate a KNN background substractor https://docs.opencv.org/3.4/de/de1/group__video__motion.html
    backgroundSubstractorObjectInstance = cv2.createBackgroundSubtractorKNN(detectShadows = False)

    # This global variable will keep track of the number of cars that go the city centre    
    global carCounter
    carCounter = 0

    # This are the region of interest position and dimensions
    # Their definition in this section of the notebook makes it easier to change its values if so wanted
    # **IMPORTANT**: making the ROI width or height small may result in the counter algorithm
    # fail to detect that the car's contour middlepoint is inside of it when analysing each of the video frames.
    # Similarly, making the ROI's width too long (bigger that the usual distance that is kept between cars when driving) 
    # may make the counting give false readings when two cars are on the same street one right next to the other.
    # A detector's width of 20 and a height of 100 had been tested to work correctly.
    # Obviously, the position and height of the ROI is crucial in avoiding to detect the other side of the road cars.
    detectorOverlayX,detectorOverlayY,detectorOverlayWidth,detectorOverlayHeight = 340, 340, 20, 100

    # This global boolean variable is used as a flag to avoid counting more than once when the same car is
    # going through inside the ROI in multiple frames. 
    global carAlreadyDetected
    carAlreadyDetected = False

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
            deltaFrame = cv2.erode(deltaFrame, None,50)        

            #The findContours function will retrieve the boundaries of the objects in the deltaFrame image
            contours,_=cv2.findContours(deltaFrame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            # Create an reactangle overlay to show in the video the section that gets ignored 
            # Overlay coude source: https://gist.github.com/IAmSuyogJadhav/305bfd9a0605a4c096383408bee7fd5c
            overlayRectangleX, overlayRectangleY, overlayRectangleWidth, overlayRectangleHeight = 3, 3, 1032, 259
            overlay = videoFrame.copy()
            cv2.rectangle(overlay, (overlayRectangleX, overlayRectangleY), (overlayRectangleX+overlayRectangleWidth, overlayRectangleY+overlayRectangleHeight), (0, 0, 0), -1)
            alpha = 0.6
            videoFrame = cv2.addWeighted(overlay, alpha, videoFrame, 1 - alpha, 0)

            # Create a transparent rectangle to visually represent the area that detects the cars going to the
            # city centre
            overlay = videoFrame.copy()
            cv2.rectangle(overlay, (detectorOverlayX, detectorOverlayY), (detectorOverlayX+detectorOverlayWidth, detectorOverlayY+detectorOverlayHeight), (0, 165, 255), -1)
            alpha = 0.6
            videoFrame = cv2.addWeighted(overlay, alpha, videoFrame, 1 - alpha, 0)

            # Create the 'Main Street' text and the rectangle that sourrounds it
            cv2.putText(videoFrame, 'Main Street', (5, 250), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255))
            cv2.line(videoFrame,(3,262), (1035,262), (0,0,255),3)
            cv2.line(videoFrame,(3,262), (3,595), (0,0,255),3)
            cv2.line(videoFrame,(1035,262), (1035,595), (0,0,255),3)
            cv2.line(videoFrame,(3,595), (1035,595), (0,0,255),3)

            # Create the 'City centre' text and the arrow that points in that direction            
            cv2.putText(videoFrame, 'City Centre', (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 165, 255))        
            videoFrame = cv2.arrowedLine(videoFrame,(200,370),(30,360),(0,165,255), 2)               

            # Iterate over the contours previously obtained.
            for contour in contours:
                # If the area of the contour is less than 3000 skip the loop (do nothing)
                # The 3000 number was obtained from testing with the video which number was the
                # minimum to avoid marking the people walking on the main street as well as the bicycles
                if cv2.contourArea(contour) < 3000:
                    continue        
                # The boundingRect function calculates the tightest rectangle that encloses the contour object
                (x, y, width, height)=cv2.boundingRect(contour)
                # If the position of the contour's enclosing rectangle is outside the main street proceed to ignore it
                if y<262:
                    continue

                # Calculate the midpoint (x and y) of the contour's bounding rectangle
                xMiddleOfContour = int((x + (x+width)) / 2) 
                yMiddleOfContour = int((y + (y+height)) / 2)            

                # This conditional will filter out the contours's middlepoint that are not inside the vertical area delimited
                # by the position where the ROI region starts (y) and where it ends (y+height)
                if yMiddleOfContour>=detectorOverlayY and yMiddleOfContour<=detectorOverlayY+detectorOverlayHeight:                            
                    # This conditional will allow the contours's middlepoint that are inside the ROI's area 
                    # (delimited by its x and x+width). Additionally, will check the flag carAlreadyDetected.
                    # If the car contour's middlepoint is inside the ROI and the flag carAlreadyDetected is false, the
                    # counter will be updated and the flag will be set to true. 
                    # The carAlreadyDetected flag will be resetted (to allow the counting of the next car that
                    # goes through the ROI) when the car leaves the ROI.
                    # Technically, "leaving the ROI" is delimited by a (non-visually represented) area that has the same width as the ROI                    
                    # but is immediately at its left. This will reset the ROI and the next car will be detected.
                    if xMiddleOfContour>=detectorOverlayX and xMiddleOfContour<=detectorOverlayX+detectorOverlayWidth and not carAlreadyDetected:                    
                        carAlreadyDetected=True
                        carCounter=carCounter+1                                        
                    elif (xMiddleOfContour+width/2)<detectorOverlayX and (xMiddleOfContour+width/2)>(detectorOverlayX-detectorOverlayWidth):                    
                        carAlreadyDetected = False

                    # Only for aesthetic purposes, show the contour rectangle and the middle point when 
                    # the car is close to the overlay counter region, these lines can be removed if so wanted
                    if (xMiddleOfContour<=detectorOverlayX and xMiddleOfContour>=detectorOverlayX-detectorOverlayWidth) or (xMiddleOfContour>=detectorOverlayX and xMiddleOfContour<=detectorOverlayX+(detectorOverlayWidth*2)):
                        cv2.circle(videoFrame, (xMiddleOfContour, yMiddleOfContour), 1, (255,255,255), 2)
                        # The contour's enclosing rectangle is inside the area of the main street            
                        cv2.rectangle(videoFrame, (x, y), (x+width, y+height), (255,255,255), 3)

            # Show the dynamic text for car counting
            cv2.putText(videoFrame, 'Counted cars to city centre: {carCounter}'.format(carCounter=carCounter), (detectorOverlayX, 250), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 165, 255))

            # Name the playback window and set its dimensions.
            cv2.namedWindow("Output video", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Output video", 1920, 1080)         

            # Show the output video that detects the moving cars
            cv2.imshow('Output video', videoFrame)

            # Press 'q' to 'quit'
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        else:
            # As mentioned above, this will get executed when the frame could not be read
            break    
    # When the video playback came to an end or the user pressed 'q', proceed to release the videoCapture object and close the window
    videoCaptureObject.release()
    cv2.destroyAllWindows()
    # Print the results to STDOUT
    print("\nResult:")
    print('For the video {videoPath} the total number of cars going to the city centre are {carCounter}'.format(videoPath=videoPath, carCounter=carCounter))
    
    # Calculate the number of cars that go to the city center per minute based on the length of the video
    numberOfSecondsInAMinute = 60
    carsPerMinute = carCounter/(videoPlaybackLengthInSeconds/numberOfSecondsInAMinute)
    return [carCounter,carsPerMinute]

# %%
videoPathsList = ['Traffic_Sample_1.mp4','Traffic_Sample_2.mp4']
carCountResultsList = []
for videoPath in videoPathsList:
    carCountResultsList.append(calculateCityCentreTrafficFromVideoPathAndShowVideo(videoPath))    

# %%
import pandas
# Create a pandas dataframe based on the results
resultDataframe = pandas.DataFrame(carCountResultsList, columns = ['Total number of cars', 'Cars per minute'], index=videoPathsList)

# Showing the table of results in HTML
from IPython.display import display, HTML
display(HTML(resultDataframe.to_html()))


