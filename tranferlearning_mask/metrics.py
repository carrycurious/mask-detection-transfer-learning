import numpy as np
import torch

def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def compute_iou_matrix(pred_boxes, target_boxes):
    ious = np.zeros((len(pred_boxes), len(target_boxes)), dtype=float)
    for i, p in enumerate(pred_boxes):
        for j, t in enumerate(target_boxes):
            ious[i, j] = box_iou(p, t)
    return ious


def compute_f2_score(true_labels, pred_labels, beta=2.0):
    true = np.asarray(true_labels, dtype=int)
    pred = np.asarray(pred_labels, dtype=int)
    tp = int(((true == 1) & (pred == 1)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    if tp == 0:
        return 0.0
    beta2 = beta ** 2
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)


def average_precision(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    if len(indices) == 0:
        return 0.0
    return np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])


def compute_average_precision(predictions, ground_truths, iou_threshold=0.5):
    if len(predictions) == 0:
        return 0.0
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
    gt_map = {}
    for idx, gts in enumerate(ground_truths):
        gt_map[idx] = {
            "boxes": [gt["box"] for gt in gts],
            "matched": [False] * len(gts),
        }

    tp = np.zeros(len(predictions), dtype=float)
    fp = np.zeros(len(predictions), dtype=float)
    total_gts = sum(len(gts) for gts in ground_truths)

    for i, pred in enumerate(predictions):
        image_id = pred["image_id"]
        best_iou = 0.0
        best_j = -1
        for j, gt_box in enumerate(gt_map[image_id]["boxes"]):
            if gt_map[image_id]["matched"][j]:
                continue
            if pred["class"] != ground_truths[image_id][j]["class"]:
                continue
            iou = box_iou(pred["box"], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0:
            tp[i] = 1.0
            gt_map[image_id]["matched"][best_j] = True
        else:
            fp[i] = 1.0

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    precisions = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-8)
    recalls = cumulative_tp / (total_gts + 1e-8)
    return average_precision(recalls, precisions)


def compute_map(predictions, ground_truths, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    ap_values = [compute_average_precision(predictions, ground_truths, t) for t in iou_thresholds]
    return float(np.mean(ap_values)), [float(v) for v in ap_values]
