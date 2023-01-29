import cv2  # imported lib

# loaded predtrained dataset
trained_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# imported the image->img = cv2.imread('/Users/nishantsharma/Desktop/codes/vgp.jpg')

# use live video
img = cv2.VideoCapture(0)

while True:
    sucessful_read, frame = img.read()

    # covert greyscale
    grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect face coordinates
    face_cord = trained_data.detectMultiScale(grey_img)

    # making a rectnagle or square on face format--(image,coordinatetop,cordbottom,color,thickness)
    for (x, y, w, h) in face_cord:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 153), 2)

    # display image
    cv2.imshow('photo:', frame)

    # wait for code to execute
    key = cv2.waitKey(1)
    # we can quit by pressing Q
    if key == 113 or key == 81:
        break
img.release()
