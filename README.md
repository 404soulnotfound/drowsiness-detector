# Drowsiness and Distraction Detector. 
It is a real time driver monitoring system that uses facial landmark detection to identify signs of drowsiness, yawning and distraction(head tilt). It avoids accidents on road as it triggers visual alerts.
# Setup 
Pre requisites- Python 3.8 or higher, pip, a webcam
Clone the repository
Create a virtual environment
Install dependencies
 # How it works
 When the eye is open EAR approx 0.30+. When it closes EAR drops to 0. So if EAR stays below 0.22 for 20+ frames consecutively a drowsiness alert is triggered.
 if MAR exceeds 0.6 yawn is detected
 The angle between the nose tip and chin landmarks is computed. If the angle exceeds 25 degrees a head tilt warning is fired.
 # Libraries used
 OpenCV MediaPipe NumPy
 # How to stop 
 press Q on the video window
