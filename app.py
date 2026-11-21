
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
st.write(
    "Computer Graphics and Image Processing Project"
)

st.markdown("""
Detects and tracks objects in uploaded videos using YOLO and ByteTrack.
Each confirmed tracking ID is counted only once.
""")

# ---------------- SETTINGS ----------------
CONFIDENCE = 0.35
IMAGE_SIZE = 640
MIN_TRACK_HITS = 3

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

# ---------------- VIDEO UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("Start Detection and Tracking", type="primary"):

        input_path = None
        output_path = None

        try:
            # Save uploaded video
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_input:
                temp_input.write(uploaded_file.read())
                input_path = temp_input.name

            # Open video
            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error("Unable to open the uploaded video.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                fps = 25.0

            if width <= 0 or height <= 0:
                st.error("Invalid video dimensions.")
                cap.release()
                st.stop()

            # Create output video
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_output:
                output_path = temp_output.name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            if not writer.isOpened():
                st.error("Unable to create the output video.")
                cap.release()
                st.stop()

            # Load YOLO model
            model = load_model()

            # Store tracking information
            track_information = {}

            frame_count = 0

            progress = st.progress(0)
            status = st.empty()

            st.info("Processing video. Please wait...")

            # ---------------- PROCESS VIDEO ----------------
            while True:

                success, frame = cap.read()

                if not success:
                    break

                frame_count += 1

                # YOLO + ByteTrack
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=CONFIDENCE,
                    imgsz=IMAGE_SIZE,
                    verbose=False
                )

                result = results[0]

                # Read tracking information
                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                ):

                    boxes = result.boxes
                    track_ids = boxes.id.int().cpu().tolist()
                    class_ids = boxes.cls.int().cpu().tolist()

                    for track_id, class_id in zip(
                        track_ids,
                        class_ids
                    ):

                        class_name = model.names[class_id]

                        if track_id not in track_information:
                            track_information[track_id] = {
                                "hits": 0,
                                "classes": Counter()
                            }

                        track_information[track_id]["hits"] += 1

                        track_information[track_id]["classes"][
                            class_name
                        ] += 1

                # Draw boxes and tracking IDs
                annotated_frame = result.plot()

                writer.write(annotated_frame)

                # Update progress
                if total_frames > 0:
                    progress.progress(
                        min(frame_count / total_frames, 1.0)
                    )

                status.text(
                    f"Processed {frame_count} / {total_frames} frames"
                )

            # Release video resources
            cap.release()
            writer.release()

            progress.progress(1.0)
            status.success("Video processing completed!")

            # ---------------- FINAL COUNTS ----------------
            confirmed_tracks = []

            for track_id, information in track_information.items():

                # Ignore very short-lived tracks
                if information["hits"] >= MIN_TRACK_HITS:

                    # Assign the most frequently detected class
                    final_class = information["classes"].most_common(1)[0][0]

                    confirmed_tracks.append({
                        "Tracking ID": track_id,
                        "Object": final_class,
                        "Frames Detected": information["hits"]
                    })

            tracking_df = pd.DataFrame(confirmed_tracks)

            if not tracking_df.empty:

                object_counts = (
                    tracking_df["Object"]
                    .value_counts()
                    .rename_axis("Object")
                    .reset_index(name="Unique Tracked IDs")
                )

                total_objects = len(tracking_df)

            else:

                object_counts = pd.DataFrame(
                    columns=["Object", "Unique Tracked IDs"]
                )

                total_objects = 0

            # ---------------- DASHBOARD ----------------
            st.header("📊 Detection Dashboard")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Frames", frame_count)
            col2.metric("Confirmed Tracked Objects", total_objects)
            col3.metric("Object Categories", len(object_counts))

            st.subheader("Object Counts")

            if not object_counts.empty:
                st.dataframe(
                    object_counts,
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    object_counts.set_index("Object")
                )

                st.subheader("Tracking Details")

                st.dataframe(
                    tracking_df,
                    use_container_width=True,
                    hide_index=True
                )

                csv_data = tracking_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Tracking Report (CSV)",
                    data=csv_data,
                    file_name="object_tracking_report.csv",
                    mime="text/csv"
                )

            else:
                st.warning(
                    "No objects were tracked for at least "
                    f"{MIN_TRACK_HITS} frames."
                )

            # ---------------- PROCESSED VIDEO ----------------
            st.header("🎬 Processed Video")

            with open(output_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(video_bytes)

            st.download_button(
                "Download Processed Video",
                data=video_bytes,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

            st.success("Analysis completed successfully!")

        except Exception as error:
            st.error(f"An error occurred: {error}")

        finally:
            # Release and remove temporary input file
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
