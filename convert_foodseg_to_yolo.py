#!/usr/bin/env python3
"""Convert FoodSeg103 bitmap annotations to YOLO segmentation format."""
import base64
import io
import json
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_class_mapping(meta_path: Path) -> tuple[list[str], dict[int, int]]:
    """Load class names and create classId -> YOLO index mapping."""
    with open(meta_path) as f:
        meta = json.load(f)
    classes = [c["title"] for c in meta["classes"]]
    id_to_idx = {c["id"]: i for i, c in enumerate(meta["classes"])}
    return classes, id_to_idx


def decode_bitmap_to_mask(bitmap_dict: dict, img_height: int, img_width: int) -> np.ndarray | None:
    """Decode base64 zlib-compressed bitmap and place it on full-size canvas at origin."""
    try:
        data_b64 = bitmap_dict["data"]
        origin_x, origin_y = bitmap_dict["origin"]
        
        # Decode: base64 -> zlib decompress -> PNG
        compressed = base64.b64decode(data_b64)
        decompressed = zlib.decompress(compressed)
        pil_img = Image.open(io.BytesIO(decompressed))
        
        # Convert to numpy array
        local_img = np.array(pil_img)
        
        if local_img is None or local_img.size == 0:
            return None
        
        # Handle different image modes
        if pil_img.mode == 'P':  # Palette mode
            local_mask = (local_img > 0).astype(np.uint8) * 255
        elif pil_img.mode == 'RGBA':
            local_mask = (local_img[:, :, 3] > 0).astype(np.uint8) * 255
        elif pil_img.mode == 'L':  # Grayscale
            local_mask = (local_img > 0).astype(np.uint8) * 255
        elif len(local_img.shape) == 3:
            local_mask = (local_img[:, :, 0] > 0).astype(np.uint8) * 255
        else:
            local_mask = (local_img > 0).astype(np.uint8) * 255
        
        # Create full-size canvas and place mask at origin
        full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        h, w = local_mask.shape[:2]
        y_end = min(origin_y + h, img_height)
        x_end = min(origin_x + w, img_width)
        full_mask[origin_y:y_end, origin_x:x_end] = local_mask[:y_end-origin_y, :x_end-origin_x]
        
        return full_mask
    except Exception as e:
        return None


def mask_to_yolo_polygon(mask: np.ndarray, img_width: int, img_height: int) -> list[float]:
    """Convert binary mask to YOLO polygon format (normalized coordinates)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Use largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Simplify polygon
    epsilon = 0.001 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) < 3:
        return []
    
    # Flatten and normalize
    coords = []
    for point in approx.reshape(-1, 2):
        x_norm = point[0] / img_width
        y_norm = point[1] / img_height
        coords.extend([x_norm, y_norm])
    
    return coords


def process_annotation(ann_path: Path, img_dir: Path, out_label_dir: Path, id_to_idx: dict[int, int]) -> None:
    """Convert single annotation JSON to YOLO segmentation label."""
    with open(ann_path) as f:
        ann = json.load(f)
    
    img_height = ann["size"]["height"]
    img_width = ann["size"]["width"]
    objects = ann.get("objects", [])
    
    if not objects:
        return
    
    out_label_dir.mkdir(parents=True, exist_ok=True)
    # ann_path.stem yields "00000000.jpg"; strip the extra extension for label name
    label_stem = Path(ann_path.stem).stem
    out_label_path = out_label_dir / f"{label_stem}.txt"
    
    lines = []
    skipped = 0
    for obj in objects:
        if obj.get("geometryType") != "bitmap" or "bitmap" not in obj:
            continue
        
        class_id = obj.get("classId")
        if class_id not in id_to_idx:
            continue
        
        cls_idx = id_to_idx[class_id]
        
        # Decode bitmap to full-size mask
        mask = decode_bitmap_to_mask(obj["bitmap"], img_height, img_width)
        
        if mask is None:
            skipped += 1
            continue
        
        # Convert mask to polygon
        polygon = mask_to_yolo_polygon(mask, img_width, img_height)
        
        if polygon and len(polygon) >= 6:  # At least 3 points (x,y pairs)
            coords_str = " ".join(f"{c:.6f}" for c in polygon)
            lines.append(f"{cls_idx} {coords_str}")
    
    if lines:
        out_label_path.write_text("\n".join(lines))
    
    if skipped > 0:
        print(f"  └─ Skipped {skipped} corrupted bitmap(s)")


def process_split(split: str, ds_root: Path, out_root: Path, id_to_idx: dict[int, int]) -> None:
    """Process train or test split."""
    img_dir = ds_root / split / "img"
    ann_dir = ds_root / split / "ann"
    out_label_dir = out_root / "labels" / split
    out_img_dir = out_root / "images" / split
    
    # Create symlinks for images
    out_img_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            link_path = out_img_dir / img_path.name
            if not link_path.exists():
                link_path.symlink_to(img_path.resolve())
    
    # Process annotations
    ann_files = sorted(ann_dir.glob("*.json"))
    for i, ann_path in enumerate(ann_files, 1):
        print(f"Processing {split} {i}/{len(ann_files)}: {ann_path.name}")
        try:
            process_annotation(ann_path, img_dir, out_label_dir, id_to_idx)
        except Exception as e:
            print(f"Error processing {ann_path.name}: {e}")


def create_data_yaml(out_root: Path, class_names: list[str]) -> None:
    """Create YOLO data.yaml configuration file."""
    yaml_content = f"""# FoodSeg103 dataset configuration for YOLO segmentation training
path: {out_root.resolve()}
train: images/train
val: images/test

# Classes
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nCreated {yaml_path}")


def main():
    ds_root = Path("dataset/foodseg103-DatasetNinja")
    out_root = Path("formatted_dataset/foodseg103-yolo")
    meta_path = ds_root / "meta.json"
    
    print("Loading class mapping...")
    class_names, id_to_idx = load_class_mapping(meta_path)
    print(f"Found {len(class_names)} classes")
    
    for split in ["train", "test"]:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print('='*60)
        process_split(split, ds_root, out_root, id_to_idx)
    
    create_data_yaml(out_root, class_names)
    print(f"\n✓ Done! Labels saved to {out_root / 'labels'}")
    print(f"\nTo train: yolo segment train data={out_root / 'data.yaml'} model=yolo26n-seg.pt epochs=100")


if __name__ == "__main__":
    main()
