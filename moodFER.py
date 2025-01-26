from fer import FER
import cv2

# Open the webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for the webcam or provide the video file path
detector = FER(mtcnn=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze emotions in the current frame
    result = detector.detect_emotions(frame)

    if result:
        emotion, score = detector.top_emotion(frame)
        label = f"Emotion: {emotion}, Score: {score:.2f}"
    else:
        label = "No face detected."

    # Display the emotions on the frame
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Driver Emotion Detection", frame)

    # Press 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

