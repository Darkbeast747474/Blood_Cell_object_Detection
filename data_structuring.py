import os
import random
import shutil
import xml.etree.ElementTree as ET

# Paths
input_image_dir = "BCCD/JPEGImages"
input_xml_dir = "BCCD/Annotations"
output_dir = "dataset"

class_labels = ["WBC", "RBC", "Platelets"]

# YOLO dataset structure
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)


def convert_bbox(size, box):
    """Converts bounding box from XML format to YOLO format."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0 * dw
    y_center = (box[2] + box[3]) / 2.0 * dh
    width = (box[1] - box[0]) * dw
    height = (box[3] - box[2]) * dh
    return (x_center, y_center, width, height)


def parse_xml(xml_file):
    """Parses an XML file and returns annotations in YOLO format."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    yolo_annotations = []

    for obj in root.iter("object"):
        cls = obj.find("name").text
        if cls not in class_labels:
            continue
        cls_id = class_labels.index(cls)

        xml_box = obj.find("bndbox")
        box = (
            float(xml_box.find("xmin").text),
            float(xml_box.find("xmax").text),
            float(xml_box.find("ymin").text),
            float(xml_box.find("ymax").text),
        )
        bbox = convert_bbox((w, h), box)
        yolo_annotations.append(
            f"{cls_id} " + " ".join(f"{coord:.6f}" for coord in bbox)
        )

    return yolo_annotations


# Gather image file names and split into train, val, and test
image_files = [f for f in os.listdir(input_image_dir) if f.endswith(".jpg")]
random.shuffle(image_files)

num_train = int(0.7 * len(image_files))
num_val = int(0.2 * len(image_files))

train_files = image_files[:num_train]
val_files = image_files[num_train : num_train + num_val]
test_files = image_files[num_train + num_val :]


# Processing and moving files
def process_files(files, split):
    """Processes and moves files to YOLO directory structure."""
    for image_file in files:
        image_path = os.path.join(input_image_dir, image_file)
        xml_path = os.path.join(input_xml_dir, os.path.splitext(image_file)[0] + ".xml")

        if not os.path.exists(xml_path):
            print(f"Warning: XML file for {image_file} not found.")
            continue

        # Parse XML and save YOLO annotations
        yolo_annotations = parse_xml(xml_path)
        if not yolo_annotations:
            print(f"No valid annotations for {image_file}")
            continue

        # Save image
        shutil.copy(image_path, os.path.join(output_dir, "images", split, image_file))

        # Save YOLO label file
        yolo_label_path = os.path.join(
            output_dir, "labels", split, os.path.splitext(image_file)[0] + ".txt"
        )
        with open(yolo_label_path, "w") as label_file:
            label_file.write("\n".join(yolo_annotations))


# Process each dataset split
process_files(train_files, "train")
process_files(val_files, "val")
process_files(test_files, "test")
