import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import pandas as pd

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Streamlit app title and instructions
center_col = st.columns([1, 13, 1])[1] 
with center_col:
    st.title("Blood Cell Detection Web App")
    st.write("#### Upload an Blood Sample Image To predict the classes")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Read and display the uploaded image
    image = Image.open(uploaded_file)

    # Run inference
    results = model.predict(source=image)

    # Draw BoundBoxes With Class and Confidence Score
    annot_img = image.copy()
    draw = ImageDraw.Draw(annot_img)
    colors = {"WBC": "blue", "RBC": "red", "Platelets": "white"}
    classes,confidences = [],[]
    
    for result in results:
        for box in result.boxes.data.tolist():
            x_min, y_min, x_max, y_max, confidence, class_id = box
            classes.append(model.names[int(class_id)])
            confidences.append(confidence)
            label = f"{model.names[int(class_id)]} ({confidence:.2f})"
            color = colors.get(model.names[int(class_id)])
            draw.rectangle(((x_min, y_min), (x_max, y_max)), outline=color, width=2)
            draw.text((x_min, y_min), label, fill="white", width=2)

    df = pd.DataFrame({"Class": classes, "Confidence(Aggregated)": confidences})
    agg_df = df.groupby("Class").mean().join(df['Class'].value_counts(), on='Class')
    
    # Display images side-by-side
    col1, col2 = st.columns(2)
    col1.image(image, caption="Original Image", use_column_width=True)
    col2.image(annot_img, caption="Detected Objects", use_column_width=True)

    center_col = st.columns([1.3, 2, 1])[1]
    with center_col:
        st.write("## Detection Results")
        st.dataframe(agg_df)
    
