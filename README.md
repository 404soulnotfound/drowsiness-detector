![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=flat-square&logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

# Drowsiness and Distraction Detector. 
It is a real time driver monitoring system that uses facial landmark detection to identify signs of drowsiness, yawning and distraction(head tilt). The system detects potential fatigue and distraction indicators and provides visual alerts to support driver awareness.

# Setup 
Pre requisites- Python 3.8 or higher, pip, a webcam
Clone the repository
Create a virtual environment
Install dependencies

 # Workflow
    Webcam
   ↓
MediaPipe Face Landmarks
   ↓
┌──────────────┬──────────────┬──────────────┐
│ Eye Landmarks│ Mouth Points │ Head Geometry│
│     EAR      │     MAR      │  Tilt Angle  │
└──────────────┴──────────────┴──────────────┘
        ↓
Temporal Filtering
        ↓
Drowsiness / Yawn / Distraction
        ↓
Visual Alert


 
 # How it works
 When the eye is open EAR approx 0.30+. When it closes EAR drops to 0. So if EAR stays below 0.22 for 20+ frames consecutively a drowsiness alert is triggered.
 if MAR exceeds 0.6 yawn is detected
 The angle between the nose tip and chin landmarks is computed. If the angle exceeds 25 degrees a head tilt warning is fired.

 ## What makes this different

Most drowsiness detectors only track eye closure. This system tracks
three signals simultaneously:
- **Eye closure** — Eye Aspect Ratio (EAR) with a 20-frame temporal
  filter to eliminate false alarms from normal blinks
- **Yawning** — Mouth Aspect Ratio (MAR) detects early fatigue signs
- **Head tilt** — Geometric angle between nose and chin landmarks
  flags distraction or phone use

Built with MediaPipe (no external model files needed) instead of dlib
— faster, lighter, works on CPU at 30+ FPS.

 # Libraries used
 OpenCV MediaPipe NumPy

 # Limitations
 Thresholds are currently empirically selected rather than learned from a labeled dataset.
 Performance can vary with lighting, camera angle and partial face occlusion.
 Current system focuses on facial cues rather than physiological signals.

 
 # How to stop 
 press Q on the video window
