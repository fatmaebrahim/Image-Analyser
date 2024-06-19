import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("yolov9m.pt")
def analyse_image(image): 
    results = model(image)
    object_names = [ model.names[int(box.cls[0])] for box in results[0].boxes ]
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        object_name = model.names[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{object_name}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 255, 10), 1)
    return image, object_names
    

st.title("Image Analyser")
uploaded_image = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"], accept_multiple_files=False)
if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button("Analyse Image"):
        analysed_image, object_names = analyse_image(image)
        st.image(analysed_image, caption='Analysed Image with YOLOv9', use_column_width=True)
        st.write("### Image Components:")
        for obj in object_names:
            st.write(f"- {obj}")
        
