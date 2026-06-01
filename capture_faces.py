import cv2
import os

# Input Name and Gender
name = input("Enter Name: ").strip()
gender = input("Enter Gender (Male/Female): ").strip()

# Dataset path (name-wise folders)
save_path = f"dataset/{name}"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

count = 0
max_images = 100   # per person images

print("Camera started... Press 'q' to stop")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))

        # Save image with name label
        file_name = os.path.join(
            save_path,
            f"{name}_{gender}_{count}.jpg"
        )

        cv2.imwrite(file_name, face)

        count += 1

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"{name} | {gender} | {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Dataset created for {name} ({gender}) with {count} images")