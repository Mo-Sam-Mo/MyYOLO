import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (use yolov8n.pt for speed)
model = YOLO("MyModel.pt")

st.title("🔴 Live YOLO Object Detection ")
st.write("Real-time webcam detection using YOLO and Streamlit.")

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO detection
        results = model(img, verbose=False)[0]

        # Draw boxes
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Launch webcam stream
webrtc_streamer(
    key="live",
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
