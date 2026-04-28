#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNXRuntime inference for the proposed FLS detector."""

import argparse
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = (
    "cube", "ball", "cylinder", "human_body", "plane",
    "circle_cage", "square_cage", "metal_bucket", "tyre", "rov",
)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STRIDES = (8, 16, 32)


def make_parser():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser("ONNXRuntime inference for the proposed FLS detector")
    parser.add_argument("--onnx", default=str(script_dir / "our_method.onnx"), help="path to ONNX model")
    parser.add_argument("--input", required=True, help="path to an image or a directory of images")
    parser.add_argument("--output-dir", default=str(script_dir / "onnxruntime_vis"), help="output directory")
    parser.add_argument("--score-thr", type=float, default=0.3, help="score threshold after obj * cls")
    parser.add_argument("--nms-thr", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="ONNXRuntime provider")
    parser.add_argument("--input-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=None,
                        help="manual input size, for dynamic ONNX input shape, e.g. --input-size 416 416")
    parser.add_argument("--save-txt", action="store_true", help="save txt detection results")
    return parser


def get_image_list(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXT:
            raise ValueError(f"unsupported image extension: {path.suffix}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    return [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_EXT]


def preproc(img, input_size):
    input_h, input_w = input_size
    padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    ratio = min(input_h / img.shape[0], input_w / img.shape[1])
    new_w = max(1, int(img.shape[1] * ratio))
    new_h = max(1, int(img.shape[0] * ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    padded[:new_h, :new_w] = resized
    blob = padded.transpose(2, 0, 1).astype(np.float32)
    return np.expand_dims(np.ascontiguousarray(blob), axis=0), ratio


def make_grids_and_strides(input_size):
    input_h, input_w = input_size
    grids, expanded_strides = [], []
    for stride in STRIDES:
        hsize, wsize = input_h // stride, input_w // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2).astype(np.float32)
        grids.append(grid)
        expanded_strides.append(np.full((1, grid.shape[1], 1), stride, dtype=np.float32))
    return np.concatenate(grids, axis=1), np.concatenate(expanded_strides, axis=1)


def decode_outputs(outputs, input_size):
    grids, strides = make_grids_and_strides(input_size)
    outputs = outputs.copy()
    if outputs.ndim != 3:
        raise ValueError(f"unexpected ONNX output shape: {outputs.shape}")
    if outputs.shape[1] != grids.shape[1]:
        raise ValueError(
            "output grid size does not match input size. "
            f"output={outputs.shape}, expected grids={grids.shape}. "
            "The ONNX model may already contain decode operations."
        )
    outputs[..., 0:2] = (outputs[..., 0:2] + grids) * strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * strides
    return outputs


def cxcywh_to_xyxy(boxes):
    boxes = boxes.copy()
    x_c, y_c, w, h = boxes[:, 0].copy(), boxes[:, 1].copy(), boxes[:, 2].copy(), boxes[:, 3].copy()
    boxes[:, 0] = x_c - w * 0.5
    boxes[:, 1] = y_c - h * 0.5
    boxes[:, 2] = x_c + w * 0.5
    boxes[:, 3] = y_c + h * 0.5
    return boxes


def compute_iou(box, boxes):
    xx1, yy1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    xx2, yy2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-12)


def nms(boxes, scores, iou_thr):
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = compute_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=np.int64)


def postprocess(predictions, ratio, orig_h, orig_w, score_thr, nms_thr):
    predictions = predictions[0]
    if predictions.shape[1] < 6:
        raise ValueError(f"unexpected prediction shape: {predictions.shape}")
    class_scores = predictions[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    class_conf = class_scores[np.arange(len(class_ids)), class_ids]
    obj_conf = predictions[:, 4]
    scores = obj_conf * class_conf
    keep = scores >= score_thr
    if not np.any(keep):
        return np.zeros((0, 7), dtype=np.float32)

    boxes = cxcywh_to_xyxy(predictions[keep, :4])
    obj_conf, class_conf = obj_conf[keep], class_conf[keep]
    class_ids, scores = class_ids[keep].astype(np.float32), scores[keep]
    keep_idx = nms(boxes, scores, nms_thr)
    boxes = boxes[keep_idx] / ratio
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, orig_w - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, orig_h - 1)
    return np.concatenate([boxes, obj_conf[keep_idx, None], class_conf[keep_idx, None], class_ids[keep_idx, None]], axis=1).astype(np.float32)


def class_name(cls_id):
    return CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"class_{cls_id}"


def draw_detections(image, detections):
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det
        cls_id = int(cls_id)
        score = float(obj_conf * cls_conf)
        color = (int((37 * cls_id + 70) % 255), int((17 * cls_id + 120) % 255), int((29 * cls_id + 200) % 255))
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_name(cls_id)} {score:.3f}"
        cv2.putText(vis, label, (int(x1), max(int(y1) - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def save_txt(txt_path, detections):
    with open(txt_path, "w", encoding="utf-8") as f:
        for det in detections:
            x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det.tolist()
            score = obj_conf * cls_conf
            f.write(f"{class_name(int(cls_id))} {score:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")


def create_session(onnx_path, device):
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" and "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def resolve_input_size(session, manual_input_size=None):
    if manual_input_size is not None:
        return int(manual_input_size[0]), int(manual_input_size[1])
    shape = session.get_inputs()[0].shape
    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int):
        return int(h), int(w)
    raise ValueError("Dynamic ONNX input shape detected. Please add --input-size 416 416")


def main():
    args = make_parser().parse_args()
    image_paths = get_image_list(args.input)
    if not image_paths:
        raise FileNotFoundError(f"no images found under: {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = create_session(args.onnx, args.device)
    input_name = session.get_inputs()[0].name
    input_size = resolve_input_size(session, args.input_size)

    print(f"providers: {session.get_providers()}")
    print(f"input_name: {input_name}")
    print(f"input_size: {input_size}")
    print(f"num_images: {len(image_paths)}")
    print(f"output_dir: {output_dir}")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"skip unreadable image: {image_path}")
            continue
        blob, ratio = preproc(image, input_size)
        outputs = session.run(None, {input_name: blob})[0]
        outputs = decode_outputs(outputs, input_size)
        detections = postprocess(outputs, ratio, image.shape[0], image.shape[1], args.score_thr, args.nms_thr)
        vis = draw_detections(image, detections)
        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), vis)
        if args.save_txt:
            save_txt(output_dir / f"{image_path.stem}.txt", detections)
        print(f"{image_path} -> {out_path}  detections={len(detections)}")


if __name__ == "__main__":
    main()
