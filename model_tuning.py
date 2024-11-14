from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on dataset for 100 epochs
results = model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640
)