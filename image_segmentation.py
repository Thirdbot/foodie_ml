from ultralytics import YOLO
from pathlib import Path


def classify_and_extract_masks(results):
    """Inspect detections and collect masks with extra logging."""
    masks = []
    for result in results:
        classnames = result.names
        num_boxes = len(result.boxes)
        num_masks = 0 if result.masks is None or result.masks.data is None else result.masks.data.shape[0]
        print(f"Detections: boxes={num_boxes}, masks={num_masks}")

        if num_boxes == 0:
            print("No detections; try lower conf or check the model path.")
            continue

        if result.masks is None or result.masks.data is None:
            print("No masks returned for this image.")
            continue

        mask_data = result.masks.data  # [N, H, W] or [B, N, H, W]

        # object detection boxes and aligned masks
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            cls_name = classnames[cls_id]

            # Align detection index to mask tensor
            if mask_data.ndim == 4:
                mask_tensor = mask_data[0, i]
            elif mask_data.ndim == 3:
                mask_tensor = mask_data[i]
            else:
                raise ValueError(f"Unexpected mask shape: {tuple(mask_data.shape)}")

            masks.append((cls_name, mask_tensor))

            conf = float(box.conf[0])
            print(f"Detected {cls_name} with confidence {conf:.2f}")

    if masks:
        get_images_area(masks)
    else:
        print("No masks collected for area calculation.")

def get_images_area(mask_tensor):
    for cls_name, mask in mask_tensor:
        bin_mask = (mask.cpu() > 0.5)
        area = int(bin_mask.sum().item())
        h, w = mask.shape[-2], mask.shape[-1]
        print(f"Mask shape: {tuple(mask.shape)} (HxW: {h}x{w})")
        print(f"{cls_name} mask area (pixels): {area}")


if __name__ == "__main__":
    # steps:
    # image -> classify -> 
    # optional(areas -> weight estimation -> class + weight) 
    # -> nutrition lookup

    trained_model_path = Path('runs/segment/train3/weights/best.pt')
    yolo_segmodel = YOLO(trained_model_path, task='segment')
    image_path = Path('test_images/00000001.jpg')

    results = yolo_segmodel(image_path, save=True, conf=0.001)
    classify_and_extract_masks(results)
