import argparse
import os
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize VOC XML annotations on an image and save the result."
    )
    parser.add_argument("--xml", required=True, help="Path to VOC XML annotation.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--output",
        default="",
        help="Path to output image. Defaults to <image_stem>_voc_vis.jpg beside input image.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=3,
        help="Bounding box line width.",
    )
    return parser.parse_args()


def load_voc_objects(xml_path):
    root = ET.parse(xml_path).getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="unknown").strip()
        difficult = int(obj.findtext("difficult", default="0"))
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = int(float(bndbox.findtext("xmin", default="0")))
        ymin = int(float(bndbox.findtext("ymin", default="0")))
        xmax = int(float(bndbox.findtext("xmax", default="0")))
        ymax = int(float(bndbox.findtext("ymax", default="0")))
        objects.append(
            {
                "name": name,
                "difficult": difficult,
                "bbox": (xmin, ymin, xmax, ymax),
            }
        )
    return objects


def choose_color(label, difficult):
    palette = [
        (231, 76, 60),
        (52, 152, 219),
        (46, 204, 113),
        (241, 196, 15),
        (155, 89, 182),
        (230, 126, 34),
        (26, 188, 156),
        (149, 165, 166),
    ]
    color = palette[sum(ord(ch) for ch in label) % len(palette)]
    if difficult:
        return (128, 128, 128)
    return color


def draw_annotations(image, objects, line_width):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for obj in objects:
        xmin, ymin, xmax, ymax = obj["bbox"]
        label = obj["name"]
        if obj["difficult"]:
            label = f"{label} (difficult)"
        color = choose_color(obj["name"], obj["difficult"])

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=line_width)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = xmin
        text_y = ymin - text_h - 4
        if text_y < 0:
            text_y = ymin + 2
        draw.rectangle(
            [text_x, text_y, text_x + text_w + 4, text_y + text_h + 4],
            fill=color,
        )
        draw.text((text_x + 2, text_y + 2), label, fill=(255, 255, 255), font=font)
    return canvas


def build_output_path(image_path, output_path):
    if output_path:
        return output_path
    stem, _ = os.path.splitext(image_path)
    return f"{stem}_voc_vis.jpg"


def main():
    args = parse_args()
    objects = load_voc_objects(args.xml)
    image = Image.open(args.image)
    annotated = draw_annotations(image, objects, max(args.line_width, 1))
    output_path = build_output_path(args.image, args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    annotated.save(output_path)
    print(f"Saved annotated image to: {output_path}")
    print(f"Number of objects: {len(objects)}")


if __name__ == "__main__":
    main()
