# Lock-In Homework Detector
   Lock-In Homework detector is a simple Python application that monitors focus of a student during homework sessions. It uses Computer Vision techniques to tract face and eye movements and estimate head pose to determine if the student is engaged with their work. The system supports both external and built-in webcams and uses MediaPipe and OpenCV for reliable detection. 

## Features 
- **Face and Eye Tracking:** Utilize MediaPipe to detect facial landmarks and track eye movements for focus analysis.
- **Head Pose Estimation:** Analyzes head orientation to assess whether the student is facing their work.
- **Webcam Support:** Compatible with both built-in and external webcams for flexible setup.
- **Real-Time Monitoring:** Provides live feedback on the student's focus status.

## Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- Numpy (`numpy`)
- Webcam

## Installaton
&emsp; **1. Clone the repository:**
`git clone https://github.com/your-username/Lock-In_Homework_Detector.git`
`cd Detector` <br>
&emsp; **2. Install Depedencies** <br>
&emsp; **3. Ensure a webcam is connected and functional.**

## Usage
1. Run the Script: `python lockin.py`
2. The application will access the webcam and begin monitoring.
3. A window will display the live feed with overlays indicating face, eye, and head pose detection, along with focus status.
4. Press q to quit the application.

## How it works
- **Face Detection:** MediaPipe detects facial landmarks to locate the face and eyes.
- **Eye Tracking:** Calculates the Eye Aspect Ratio (EAR) to determine if the student is looking at the screen.
- **Head Pose:** Estimates head orientation to check if the student is facing their work.
- **Focus Assessment:** Combines data from the above to classify the student's focus level.

## Future Improvements
- **Distraction Detection:** Integration of YOLO for detecting objects like phones or books is in progress to enhance focus monitoring. (I only focused on student's attentiion and movement on a screen)

## Limitations
- Requires good lighting for accurate face detection.
- Performance depends on webcam quality and system hardware.
- Distraction detection is not yet implemented but is under active development.

## Source
Lock-In Device by juiceditup: https://youtube.com/shorts/7iqF8gcw9Ww?si=AR6bl__vV1GF-rRs

## Programmer
© 2025 Juliana Mancera

