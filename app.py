
import streamlit as st
import cv2
import pandas as pd
import tempfile
import os
from collections import defaultdict
from ultralytics import YOLO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Object Detection and Tracking",
    page_icon="🚘",
    layout="wide"
)

st.title("🚘 Video-Based Object Detection and Tracking System")
st.write(
    "Detect, track and count unique objects in uploaded videos "
    "using YOLOv8 and ByteTrack."
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.40,
    step=0.05
)

st.sidebar.info(
    "Higher confidence values may reduce false detections, "
    "but may also miss some objects."
)

# ---------------- VIDEO UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    # Save uploaded video temporarily
    input_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    input_temp.write(uploaded_file.getvalue())
    input_temp.close()

    video_path = input_temp.name

    # Video information
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30

    cap.release()

    st.video(uploaded_file)

    st.write("### Video Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Video FPS", round(fps, 2))
    col2.metric("Resolution", f"{width} x {height}")
    col3.metric("Total Frames", total_frames)

    # ---------------- START PROCESSING ----------------
    if st.button("Start Detection and Tracking", type="primary"):

        cap = cv2.VideoCapture(video_path)

        output_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )
        output_temp.close()

        output_path = output_temp.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        # Store unique tracked objects
        tracked_objects = {}

        # Store detection events across frames
        detection_events = defaultdict(int)

        progress_bar = st.progress(0)
        status_text = st.empty()
        video_display = st.empty()

        frame_count = 0

        # ---------------- PROCESS VIDEO ----------------
        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                verbose=False
            )

            result = results[0]

            # Draw bounding boxes and tracking IDs
            annotated_frame = result.plot()

            # Extract tracking IDs and classes
            if result.boxes is not None and result.boxes.id is not None:

                ids = result.boxes.id.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for track_id, class_id in zip(ids, classes):

                    class_name = model.names[int(class_id)]

                    # Count detections across frames
                    detection_events[class_name] += 1

                    # Store each unique tracking ID only once
                    if track_id not in tracked_objects:
                        tracked_objects[track_id] = class_name

            # Write processed frame
            writer.write(annotated_frame)

            # Display processed frame
            preview_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            video_display.image(
                preview_frame,
                channels="RGB",
                use_container_width=True
            )

            frame_count += 1

            # Update progress
            if total_frames > 0:
                progress_bar.progress(
                    min(frame_count / total_frames, 1.0)
                )

            status_text.text(
                f"Processing frame {frame_count}/{total_frames}"
            )

        # Release resources
        cap.release()
        writer.release()

        progress_bar.progress(1.0)
        status_text.success("Video processing completed!")

        # ---------------- DASHBOARD ----------------
        st.subheader("📊 Object Counting Dashboard")

        # Total unique tracked IDs
        total_unique_objects = len(tracked_objects)

        # Class-wise unique counts
        unique_class_counts = defaultdict(int)

        for class_name in tracked_objects.values():
            unique_class_counts[class_name] += 1

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Unique Objects",
            total_unique_objects
        )

        col2.metric(
            "Object Categories",
            len(unique_class_counts)
        )

        # ---------------- UNIQUE OBJECTS ----------------
        if unique_class_counts:

            st.subheader("Unique Objects by Category")

            df = pd.DataFrame(
                list(unique_class_counts.items()),
                columns=["Object", "Unique Count"]
            )

            df = df.sort_values(
                by="Unique Count",
                ascending=False
            ).reset_index(drop=True)

            st.dataframe(
                df,
                use_container_width=True
            )

            # Chart
            st.subheader("Unique Object Count Chart")

            st.bar_chart(
                df.set_index("Object")["Unique Count"]
            )

        else:
            st.warning("No tracked objects were detected.")

        # ---------------- DETECTION EVENTS ----------------
        st.subheader("Repeated Detection Records (Across Frames)")

        st.caption(
            "This table represents the total number of detections "
            "recorded across video frames. The same object may be "
            "detected multiple times and is not counted as a separate "
            "unique object each time."
        )

        if detection_events:

            event_df = pd.DataFrame(
                list(detection_events.items()),
                columns=["Object", "Detection Events"]
            )

            event_df = event_df.sort_values(
                by="Detection Events",
                ascending=False
            ).reset_index(drop=True)

            st.dataframe(
                event_df,
                use_container_width=True
            )

        else:
            st.info("No detection events recorded.")

        # ---------------- DOWNLOAD VIDEO ----------------
        st.subheader("📥 Processed Video")

        if os.path.exists(output_path):

            with open(output_path, "rb") as video_file:

                st.download_button(
                    label="Download Processed Video",
                    data=video_file,
                    file_name="tracked_video.mp4",
                    mime="video/mp4"
                )

        # ---------------- CLEANUP ----------------
        try:
            os.remove(video_path)
        except OSError:
            pass
