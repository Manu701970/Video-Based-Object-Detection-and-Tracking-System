
import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from collections import Counter
from ultralytics import YOLO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Object Detection and Tracking System",
    page_icon="🎯",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🎯 Video-Based Object Detection and Tracking System")
st.write("Computer Graphics and Image Processing Project")

st.sidebar.title("Project Information")
st.sidebar.write("**Project:** Video-Based Object Detection and Tracking System")
st.sidebar.write("**Model:** YOLOv8 Nano")
st.sidebar.write("**Tracker:** ByteTrack")
st.sidebar.write("**Language:** Python")
st.sidebar.write("**Libraries:** OpenCV, Streamlit, Ultralytics")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- VIDEO UPLOAD ----------------
st.subheader("Upload Video")

uploaded_file = st.file_uploader(
    "Select a video file",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("Start Detection", type="primary"):

        input_path = None
        output_path = None

        try:
            # ---------------- SAVE INPUT VIDEO ----------------
            with tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False
            ) as input_temp:

                input_temp.write(uploaded_file.getbuffer())
                input_path = input_temp.name

            # ---------------- OUTPUT FILE ----------------
            output_temp = tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False
            )

            output_path = output_temp.name
            output_temp.close()

            # ---------------- VIDEO CAPTURE ----------------
            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error("Unable to open the uploaded video.")
                st.stop()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps <= 0:
                fps = 30

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            # ---------------- PROCESSING UI ----------------
            st.subheader("Video Processing")

            progress_bar = st.progress(0)
            status_text = st.empty()
            video_preview = st.empty()

            unique_objects = set()
            event_counts = Counter()

            frame_number = 0

            # ---------------- PROCESS VIDEO ----------------
            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False
                )

                result = results[0]
                annotated_frame = result.plot()

                # ---------------- OBJECT DATA ----------------
                if result.boxes is not None:

                    for box in result.boxes:

                        class_id = int(box.cls[0])
                        object_name = model.names[class_id]

                        event_counts[object_name] += 1

                        if box.id is not None:

                            track_id = int(box.id[0])

                            unique_objects.add(
                                (object_name, track_id)
                            )

                # ---------------- WRITE VIDEO ----------------
                writer.write(annotated_frame)

                # ---------------- LIVE PREVIEW ----------------
                rgb_frame = cv2.cvtColor(
                    annotated_frame,
                    cv2.COLOR_BGR2RGB
                )

                video_preview.image(
                    rgb_frame,
                    channels="RGB",
                    use_container_width=True
                )

                frame_number += 1

                if total_frames > 0:

                    progress = min(
                        frame_number / total_frames,
                        1.0
                    )

                    progress_bar.progress(progress)

                status_text.write(
                    f"Processing frame {frame_number} "
                    f"of {total_frames}"
                )

            # ---------------- RELEASE ----------------
            cap.release()
            writer.release()

            progress_bar.progress(1.0)
            status_text.success("Video processing completed!")

            # ---------------- FINAL METRICS ----------------
            st.subheader("Detection Dashboard")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Unique Tracked Objects",
                len(unique_objects)
            )

            col2.metric(
                "Detection Events",
                sum(event_counts.values())
            )

            col3.metric(
                "Categories Detected",
                len(event_counts)
            )

            # ---------------- SUMMARY ----------------
            st.subheader("Object Summary")

            if event_counts:

                df = pd.DataFrame(
                    list(event_counts.items()),
                    columns=["Object", "Detection Events"]
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                # ---------------- CHART ----------------
                st.subheader("Detection Statistics")

                st.bar_chart(
                    df.set_index("Object")
                )

            else:

                st.warning("No objects detected.")

            # ---------------- OUTPUT VIDEO ----------------
            st.subheader("Processed Video")

            if os.path.exists(output_path):

                st.video(output_path)

                with open(output_path, "rb") as video_file:

                    st.download_button(
                        label="Download Processed Video",
                        data=video_file,
                        file_name="tracked_video.mp4",
                        mime="video/mp4"
                    )

        except Exception as e:

            st.error(f"Error: {e}")

        finally:

            # ---------------- CLEANUP INPUT ----------------
            if input_path and os.path.exists(input_path):

                try:
                    os.remove(input_path)
                except OSError:
                    pass
