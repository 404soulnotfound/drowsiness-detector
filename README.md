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

# Technical Approach

## Technical Approach

The system uses facial landmark geometry and temporal analysis to identify visual indicators of driver fatigue and distraction.
1. Face Landmark Detection
MediaPipe Face Mesh is used to detect facial landmarks from each webcam frame. These landmarks provide the coordinates required to analyze the eyes, mouth, and head orientation in real time.
2. Eye Aspect Ratio (EAR)
The Eye Aspect Ratio is calculated from key eye landmarks to estimate the degree of eye closure.

* **EAR < 0.22** indicates a potentially closed eye.
* The condition must persist for **20+ consecutive frames** before triggering a drowsiness alert, reducing false positives caused by normal blinking.
3. Mouth Aspect Ratio (MAR)
The Mouth Aspect Ratio is calculated using mouth landmarks to detect prolonged mouth opening.

* **MAR > 0.6** is used as the threshold for potential yawning.
* This provides an additional fatigue indicator alongside eye closure.

4. Head-Tilt Detection
Facial landmark geometry is used to estimate head orientation.

* A head tilt exceeding **25°** is treated as a potential distraction indicator.
* This helps detect cases where the driver's attention may be directed away from the road.

5. Temporal Filtering & Decision Logic
Individual frames are not evaluated in isolation. Eye-closure and other signals are tracked across consecutive frames to distinguish sustained fatigue indicators from short-lived movements.
The final alert logic combines:

**Eye Closure + Yawning + Head Tilt → Drowsiness / Distraction Alert**
6. Real-Time Processing
The complete pipeline runs continuously on webcam frames using OpenCV and MediaPipe, with lightweight landmark-based calculations designed for CPU-based real-time processing.

 
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
