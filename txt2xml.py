import os
import cv2
import xml.etree.ElementTree as ET

# Define input and output paths
image_dir = r"/path/to/images"
label_dir = r"/path/to/labels"
save_dir = r"/path/to/save_xml"

os.makedirs(save_dir, exist_ok=True)

# Class ID to class name mapping
class_names = {
    0: "uav",
    1: "helicopter",
    2: "vtol",
    3: "fix wing",
    4: "dragonfly",
    5: "bird"
}

def create_xml(filename, img_shape, objects):
    """
    Create a Pascal VOC XML structure
    """
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "filename").text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_shape[1])
    ET.SubElement(size, "height").text = str(img_shape[0])
    ET.SubElement(size, "depth").text = str(img_shape[2])

    for obj in objects:
        obj_elem = ET.SubElement(annotation, "object")
        ET.SubElement(obj_elem, "name").text = obj["name"]
        ET.SubElement(obj_elem, "pose").text = "Unspecified"
        ET.SubElement(obj_elem, "truncated").text = "0"
        ET.SubElement(obj_elem, "difficult").text = "0"

        bndbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(obj["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(obj["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(obj["ymax"])

    return annotation


# Iterate through all label files
for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(label_dir, label_file)
    image_name = label_file.replace(".txt", ".jpg")
    image_path = os.path.join(image_dir, image_name)

    # Skip if corresponding image does not exist
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    # Read image to obtain its dimensions
    img = cv2.imread(image_path)
    h, w, c = img.shape

    objects = []

    # Read YOLO annotation file
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        # YOLO format: class_id x_center y_center width height (normalized)
        class_id = int(parts[0])
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        box_w = float(parts[3]) * w
        box_h = float(parts[4]) * h

        # Convert to Pascal VOC format (absolute pixel coordinates)
        xmin = int(x_center - box_w / 2)
        ymin = int(y_center - box_h / 2)
        xmax = int(x_center + box_w / 2)
        ymax = int(y_center + box_h / 2)

        objects.append({
            "name": class_names[class_id],
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        })

    # Create XML tree
    xml_tree = create_xml(image_name, img.shape, objects)
    tree = ET.ElementTree(xml_tree)

    save_path = os.path.join(save_dir, label_file.replace(".txt", ".xml"))
    tree.write(save_path)

    print(f"Saved: {save_path}")

print("Conversion completed successfully.")