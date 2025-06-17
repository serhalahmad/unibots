import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)

ret, frame = cap.read()
if ret:
    cv2.imwrite("captured_image.jpg", frame)  # Save the image

cap.release()
