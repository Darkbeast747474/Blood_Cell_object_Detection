import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import albumentations as A


def parse_bounding_boxes(xml_path):
    """Parse bounding boxes from an XML file (PASCAL VOC format)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []
    coords = []
    
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    d = int(size.find("depth").text)
    coords.append([w, h, d])
    
    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        if xmax <= xmin or ymax <= ymin:
            pass
        else:
            labels.append(label)
            boxes.append([xmin, ymin, xmax, ymax])

    return boxes, labels, coords


def save_augmented_image_and_xml(
    image, bboxes, labels, coords, output_image_path, output_xml_path
):
    """Save the augmented image and update the XML file with new bounding boxes."""

    image.save(output_image_path)
    root = ET.Element("annotation")
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(coords[0][0])
    ET.SubElement(size, "height").text = str(coords[0][1])
    ET.SubElement(size, "depth").text = str(coords[0][2])
    for bbox, label in zip(bboxes, labels):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(bbox[0]))
        ET.SubElement(bndbox, "ymin").text = str(int(bbox[1]))
        ET.SubElement(bndbox, "xmax").text = str(int(bbox[2]))
        ET.SubElement(bndbox, "ymax").text = str(int(bbox[3]))

    tree = ET.ElementTree(root)
    tree.write(output_xml_path)


def augment_image(image_path, xml_path, output_image_dir, output_xml_dir):

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=15, p=0.5),
        ],bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"])
    )

    image = Image.open(image_path)
    image_np = np.array(image)  # Convert to NumPy array for Albumentations

    bboxes, labels, coords = parse_bounding_boxes(xml_path)
    print(f"image {image_path}")
    
    # Apply augmentations
    augmented = transform(image=image_np, bboxes=bboxes, category_ids=labels)
    augmented_image_np = augmented["image"]
    augmented_bboxes = augmented["bboxes"]

    # Convert back to PIL image
    augmented_image = Image.fromarray(augmented_image_np)

    # Generate output paths
    base_name = os.path.basename(image_path).replace(".jpg", "_aug.jpg")
    output_image_path = os.path.join(output_image_dir, base_name)
    output_xml_path = os.path.join(output_xml_dir, base_name.replace(".jpg", ".xml"))

    # Save augmented image and corresponding XML
    save_augmented_image_and_xml(
        augmented_image,
        augmented_bboxes,
        labels,
        coords,
        output_image_path,
        output_xml_path,
    )


# Example Usage
input_image_dir = "BCCD/JPEGImages/"
input_xml_dir = "BCCD/Annotations/"
output_image_dir = "BCCD/JPEGImages/"
output_xml_dir = "BCCD/Annotations/"
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_xml_dir, exist_ok=True)

# Loop through images for augmentation
for image_name in os.listdir(input_image_dir):
    image_path = os.path.join(input_image_dir, image_name)
    xml_path = os.path.join(input_xml_dir, image_name.replace(".jpg", ".xml"))
    augment_image(image_path, xml_path, output_image_dir, output_xml_dir)
