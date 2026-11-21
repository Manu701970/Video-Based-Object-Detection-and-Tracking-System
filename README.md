# 🎯 Video-Based Object Detection and Tracking System

## 🌐 Live Application
** the project :**  
[Video-Based Object Detection and Tracking System](https://video-based-object-detection-and-tracking-system-iw38lnm8dxiwa.streamlit.app/)

## 📌 Project Overview

The **Video-Based Object Detection and Tracking System** is an AI-powered computer vision application developed using Python. The system detects and tracks multiple objects in uploaded video files using the YOLOv8 Nano object detection model and ByteTrack tracking algorithm.

The application processes video frames, identifies objects, assigns tracking IDs, and generates a processed video with visual annotations. It also provides a dashboard displaying detection statistics and object summaries.

This project demonstrates the practical application of Artificial Intelligence, Deep Learning, and Computer Vision techniques.

## 🎯 Project Objectives

- Detect multiple objects in video files.
- Track detected objects across consecutive video frames.
- Assign unique tracking IDs to detected objects.
- Display object detection statistics.
- Generate and download processed videos.
- Provide a simple and interactive web interface.

## 🚀 Key Features

- **Real-Time Frame Processing:** Processes video frames sequentially.
- **Object Detection:** Identifies objects using YOLOv8 Nano.
- **Object Tracking:** Tracks objects using ByteTrack.
- **Unique Object Identification:** Assigns tracking IDs to detected objects.
- **Detection Dashboard:** Displays unique tracked objects, detection events, and object categories.
- **Object Statistics:** Displays object summaries and graphical statistics.
- **Video Download:** Allows users to download the processed video.
- **Web Interface:** Provides an easy-to-use interface through Streamlit.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| YOLOv8 Nano | Object detection |
| ByteTrack | Object tracking |
| OpenCV | Video processing |
| Streamlit | Web application interface |
| Pandas | Data processing and statistics |
| GitHub | Source code management |
| Streamlit Community Cloud | Application deployment |

## ⚙️ System Workflow

1. Upload a video file through the Streamlit application.
2. Open and read video frames using OpenCV.
3. Detect objects in each frame using YOLOv8 Nano.
4. Track detected objects using ByteTrack.
5. Assign tracking IDs to identified objects.
6. Generate annotated video frames.
7. Display detection statistics and object summaries.
8. Download the processed video.

## 📂 Project Structure

```text
Video-Based-Object-Detection-and-Tracking-System/
│
├── app.py
├── requirements.txt
└── README.md
```

The YOLOv8 Nano model (`yolov8n.pt`) is downloaded automatically by Ultralytics when required.

## 💻 Installation and Setup

### Prerequisites
- Python 3.10 or later
- Visual Studio Code
- Internet connection

### Step 1: Clone the Repository

```bash
git clone https://github.com/Manu701970/Video-Based-Object-Detection-and-Tracking-System.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Video-Based-Object-Detection-and-Tracking-System
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Expected Output

The application provides:

- Object detection with bounding boxes.
- Tracking IDs for detected objects.
- Video processing progress.
- Unique tracked object count.
- Total detection events.
- Detected object categories.
- Object summary table.
- Detection statistics chart.
- Processed video preview and download.

## 🎓 Project Information

**Project Title:** Video-Based Object Detection and Tracking System  
**Subject:** Computer Graphics and Image Processing (CGIP)  
**Programming Language:** Python  
**Detection Model:** YOLOv8 Nano  
**Tracking Algorithm:** ByteTrack  

## 🔗 Project Links

- **Live Application:** https://video-based-object-detection-and-tracking-system-iw38lnm8dxiwa.streamlit.app/
- **GitHub Repository:** https://github.com/Manu701970/Video-Based-Object-Detection-and-Tracking-System

## 👨‍💻 Author

Developed as an academic project for Computer Graphics and Image Processing (CGIP).

---

⭐ If you find this project useful, consider giving the repository a star!
