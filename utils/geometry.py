import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def assign_targets_to_anchors(gt_boxes, anchors, iou_threshold=0.5):
    B, M, _ = gt_boxes.shape
    N = anchors.size(0)

    valid_mask = gt_boxes.abs().sum(dim=-1) > 0       # [B, M]
    gt_flat = gt_boxes.view(B * M, 4)               # [B*M, 4]
    ious_flat = compute_iou(gt_flat, anchors)         # [B*M, N]
    ious = ious_flat.view(B, M, N)               # [B, M, N]

    ious = ious.masked_fill(~valid_mask.unsqueeze(-1), float('-inf'))
    max_ious, max_idxs = ious.max(dim=1)              # both [B, N]

    object_mask = (max_ious > iou_threshold).float()  # [B, N]
    batch_idx = torch.arange(B, device=gt_boxes.device).unsqueeze(1).expand(-1, N)
    target_boxes = gt_boxes[batch_idx, max_idxs]       # [B, N, 4]
    
    return target_boxes, object_mask  # [B, N, 4], [B, N]


def compute_iou(boxes1, boxes2, mode="iom", eps=1e-8):
    M, N = boxes1.size(0), boxes2.size(0)

    # boxes1: [M, 4], boxes2: [N, 4]
    area1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
    area2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])

    # intersection
    lt = torch.max(boxes1[:,None,:2], boxes2[:,:2])   # [M, N, 2]
    rb = torch.min(boxes1[:,None,2:], boxes2[:,2:])   # [M, N, 2]

    wh = (rb - lt).clamp(min=0)                        # [M, N, 2]
    inter = wh[:,:,0] * wh[:,:,1]                     # [M, N]

    if mode == "iou":
        union = area1[:,None] + area2[None,:] - inter
        return inter / (union + eps)

    elif mode == "ioa":
        return inter / (area1[:,None] + eps)

    elif mode == "iog":
        return inter / (area2[None,:] + eps)

    elif mode == "iom":
        min_area = torch.min(area1[:,None], area2[None,:])
        return inter / (min_area + eps)

    else:
        raise ValueError(f"Unknown overlap mode: {mode}")


def encode_boxes_to_offsets(target_boxes, anchors):
    # target_boxes: [B, N, 4], anchors: [N, 4]
    anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2  # [N,2]
    anchor_sizes =  anchors[:, 2:] - anchors[:, :2]      # [N,2]

    # expand to batch size
    B = target_boxes.size(0)
    anchor_centers = anchor_centers.unsqueeze(0).expand(B, -1, -1)
    anchor_sizes = anchor_sizes.unsqueeze(0).expand(B, -1, -1)

    target_centers = (target_boxes[..., :2] + target_boxes[..., 2:]) / 2
    target_sizes = target_boxes[..., 2:] - target_boxes[..., :2]
    
    offsets = torch.zeros_like(target_boxes)
    offsets[..., :2] = (target_centers - anchor_centers) / anchor_sizes
    offsets[..., 2:] = torch.log((target_sizes + 1e-8) / anchor_sizes)

    return offsets


def decode_offsets_to_boxes(offsets, anchors):
    anchors = anchors.to(offsets.device)
    
    anchor_centers = (anchors[..., :2] + anchors[..., 2:]) / 2
    anchor_sizes = anchors[..., 2:] - anchors[..., :2]

    pred_centers = offsets[..., :2] * anchor_sizes + anchor_centers
    pred_sizes = torch.exp(offsets[..., 2:]) * anchor_sizes

    boxes = torch.cat([
        pred_centers - pred_sizes / 2,
        pred_centers + pred_sizes / 2
    ], dim=-1)
    
    return boxes  # [B, N, 4]


def resize_image_and_boxes(image, boxes, size):
    """
    image: Tensor[C, H, W]
    boxes: Tensor[N, 4] in (x_min, y_min, x_max, y_max)
    size:   (new_h, new_w)
    """
    _, orig_h, orig_w = image.shape
    new_h, new_w = size

    # Resize image
    image = TF.resize(image, [new_h, new_w])

    # Compute scale factors
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    # Adjust boxes
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y

    boxes[:, [4, 6]] = boxes[:, [4, 6]] * scale_x
    boxes[:, [5, 7]] = boxes[:, [5, 7]] * scale_y

    return image, boxes


    
# __all__ = ["assign_targets_to_anchors","focal_loss","encode_boxes_to_offsets","decode_offsets_to_boxes"]