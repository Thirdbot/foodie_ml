import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import PIL.Image as Image
import cv2
import numpy as np


if __name__ == "__main__":

    # trained_model_path = Path('runs/segment/train3/weights/best.pt')
    trained_model_path = Path('yolo26n-seg.pt')
    yolo_segmodel = YOLO(trained_model_path, task='segment')

    st.title("Real-Time Food Segmentation")

    # Radio buttons for different input modes
    mode = st.radio("Choose Input Mode:", ["Real-Time Webcam", "Upload Image"], horizontal=True)

    if mode == "Real-Time Webcam":
        ## Enable/disable real-time
        enable_realtime = st.checkbox("Enable Real-Time Detection", value=True)

        # Placeholders for video feed and info
        video_placeholder = st.empty()
        info_placeholder = st.empty()

        if enable_realtime:
            cap = cv2.VideoCapture(0)  # Open default webcam
        
            if not cap.isOpened():
                st.error("Cannot open webcam")
            else:
                st.info("Real-time detection running... Press Ctrl+C to stop")
                
                while enable_realtime:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from webcam")
                        break
                    
                    # Run inference
                    results = yolo_segmodel(frame, conf=0.1, verbose=False)
                    
                    # Get annotated frame
                    annotated_frame = results[0].plot()
                    
                    # Convert BGR to RGB for Streamlit
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    video_placeholder.image(annotated_frame_rgb, channels="RGB", width=700)
                    
                    # Show detection count
                    if results[0].masks is not None:
                        num_detections = results[0].masks.data.shape[0]
                        info_placeholder.info(f"Detections: {num_detections}")
            
            cap.release()

    else:
        st.subheader("Upload Image for Segmentation")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            # Load and display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, width=700)
            
            # Run segmentation
            with st.spinner("Running segmentation..."):
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                results = yolo_segmodel(image_cv, conf=0.001, verbose=False)
                annotated_frame = results[0].plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Segmentation Results")
                st.image(annotated_frame_rgb, width=700)
            
            # Show detection info
            if results[0].masks is not None:
                num_detections = results[0].masks.data.shape[0]
                st.success(f"Found {num_detections} food items")
                
                # Show class info
                classnames = results[0].names
                for i, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls[0])
                    cls_name = classnames[cls_id]
                    conf = float(box.conf[0])
                    st.write(f"  • {cls_name}: {conf:.2%} confidence")
            else:
                st.info("No food items detected in this image")