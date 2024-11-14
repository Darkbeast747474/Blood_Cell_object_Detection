import xml.etree.ElementTree as ET
import os
from PIL import Image, ImageDraw

def draw_bounding_boxes(image_path, xml_path, annotated_dir, image_name):
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    colors = {"WBC": "blue", "RBC": "red", "Platelets": "white"}
    # Loop through each object in the XML
    for obj in root.findall('object'):
        # Get class label
        label = obj.find('name').text

        # Get bounding box coordinates
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # Draw bounding box
        color = colors.get(label)
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)
        # Draw label
        draw.text((xmin, ymin), label, fill=color,width=2)

    # Save the annotated image
    image.save(os.path.join(annotated_dir, image_name))
    return image

# Directory paths
image_dir = "BCCD/JPEGImages/"
xml_dir = "BCCD/Annotations/"

def annot_and_save(image_dir, xml_dir):
    # Create the annotated directory if it does not exist
    os.makedirs("BCCD/Preprocessed_imgs/", exist_ok=True)

    # Loop through each image and corresponding XML
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        xml_name = image_name.replace(".jpg", ".xml")
        xml_path = os.path.join(xml_dir, xml_name)
        
        draw_bounding_boxes(image_path, xml_path, "BCCD/Preprocessed_imgs/", image_name)

annot_and_save(image_dir, xml_dir)