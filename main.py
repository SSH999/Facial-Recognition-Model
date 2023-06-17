import cv2

# Load the image using OpenCV
image_path = 'foto1.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale for processing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection using a pre-trained face detector (e.g., HOG + SVM)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Iterate over each detected face
for (x, y, w, h) in faces:
    # Define the region of interest for the face
    face_roi = gray[y:y + h, x:x + w]

    # Perform eye detection using a pre-trained eye detector (e.g., Haar cascade classifier)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over each detected eye and draw a circle around it
    for (ex, ey, ew, eh) in eyes:
        center_x = x + ex + ew // 2
        center_y = y + ey + eh // 2
        radius = min(ew, eh) // 2
        cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)

    # Define the region of interest for the mouth
    mouth_roi = gray[y + 4*h//5:y + h, x:x + w]

    # Find contours in the mouth region
    contours, _ = cv2.findContours(mouth_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over each contour and fit a circle around it
    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) < 100:
            continue

        # Fit a circle around the contour
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        center_x = int(center_x) + x
        center_y = int(center_y) + y + 4*h//5
        radius = int(radius * 0.3)  # Adjust the scaling factor here to make the circle smaller

        # Draw the circle around the mouth
        cv2.circle(image, (center_x, center_y), radius, (0, 0, 255), 2)

# Display the image with bounding boxes
cv2.imshow('Facial Features', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
