import cv2

def define_detector():
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    minRepeatability = 2;

    params.filterByColor = True
    params.blobColor = 255

    params.minDistBetweenBlobs  = 10;
    # # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 255
    # params.thresholdStep = 10


    # # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 100000

    # # Filter by Circularity
    params.filterByCircularity = False
    # params.minCircularity = 0.5

    # # Filter by Convexity
    params.filterByConvexity = False
    # params.minConvexity = 0.7

    # # Filter by Inertia
    params.filterByInertia = False
    # params.minInertiaRatio = 0.1


    return cv2.SimpleBlobDetector_create(params)
    # keypoints = detector.detect((img>thr).astype(np.uint8)*255)