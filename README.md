# Gender Detection using Face Capture

## Project Description

Gender Detection using Face Capture is a computer vision and deep learning project that captures face images using a webcam and predicts the gender as Male or Female. This project uses OpenCV for face capture and image processing, TensorFlow/Keras for model training, and Streamlit for creating a simple web interface.

The main aim of this project is to detect a person's face through the camera, process the captured image, and predict gender using a trained deep learning model.

## Features

* Captures face images using webcam
* Stores captured images in dataset folders
* Trains a deep learning model using face images
* Predicts gender as Male or Female
* Uses OpenCV for face detection and preprocessing
* Provides a user-friendly Streamlit interface
* Beginner-friendly computer vision project

## Technologies Used

* Python
* OpenCV
* TensorFlow
* Keras
* Streamlit
* NumPy
* Pillow

## Project Structure

```text
Gender-Detection-Project/
│
├── dataset/
│   ├── Male/
│   └── Female/
│
├── capture_dataset.py
├── train_model.py
├── app.py
├── gender_model.h5
├── requirements.txt
└── README.md
```

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Gender-Detection-Project.git
cd Gender-Detection-Project
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 3. Capture Face Dataset

Run the following command to capture face images using webcam:

```bash
python capture_dataset.py
```

Captured images will be stored inside the dataset folder under Male and Female categories.

### 4. Train the Model

After capturing the dataset, train the gender detection model:

```bash
python train_model.py
```

After training, the model file will be saved as:

```text
gender_model.h5
```

### 5. Run the Streamlit App

To start the gender detection web application, run:

```bash
python -m streamlit run app.py
```

Then open the local URL in the browser:

```text
http://localhost:8501
```

## Project Workflow

```text
Face Capture → Image Preprocessing → Model Training → Gender Prediction → Streamlit Output
```

## Output

The application captures the user's face through the webcam and predicts the gender as:

```text
Male
```

or

```text
Female
```

## Use Case

This project can be used as a beginner-level computer vision application to understand face detection, image classification, webcam integration, and deep learning model deployment using Streamlit.

## Conclusion

The Gender Detection using Face Capture project successfully captures face images, trains a deep learning model, and predicts gender through a Streamlit web interface. It is a useful project for learning computer vision, OpenCV, and deep learning concepts.

## Author

Gowri Mahanthi
<img width="1920" height="1080" alt="Screenshot (22)" src="https://github.com/user-attachments/assets/4e0f3227-53e4-477c-ae48-dcc18da292c8" />
