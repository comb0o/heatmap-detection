import torch
import torchvision.transforms.functional as TF
from torchvision.ops import nms
from utils.model import AnchorGenerator
from utils.geometry import decode_offsets_to_boxes


def preprocess_pil(img_pil, device, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    # Convert PIL image to normalized torch tensor [1, C, H, W]
    img = TF.to_tensor(img_pil).to(device)
    img = TF.normalize(img, mean=mean, std=std)
    return img.unsqueeze(0)


def generate_anchors_for_image(model, image_size):
    # Generate anchors for the provided image_size using model.stride and model.scales
    ag = AnchorGenerator(stride=model.stride, scales=model.scales)
    return ag.generate(image_size).to(next(model.parameters()).device)


def postprocess_predictions(offsets, logits, anchors, score_thresh=0.5, iou_thresh=0.5, topk=1000):
    # Turn raw outputs into final boxes + scores per image in batch
    device = offsets.device
    scores = torch.sigmoid(logits)  # [B, N]
    B = scores.shape[0]
    results = []
    for b in range(B):
        sc = scores[b]
        keep_inds = torch.nonzero(sc >= score_thresh).squeeze(1)
        if keep_inds.numel() == 0:
            results.append({'boxes': torch.empty((0,4), device=device), 'scores': torch.empty((0,), device=device)})
            continue
        offs = offsets[b, keep_inds]     # [M,4]
        anc = anchors[keep_inds]         # [M,4]
        boxes = decode_offsets_to_boxes(offs.unsqueeze(0), anc).squeeze(0)  # [M,4]
        sc = sc[keep_inds]

        # optionally restrict candidates before NMS
        if boxes.size(0) > topk:
            sc_topk_vals, sc_topk_idx = torch.topk(sc, topk)
            boxes = boxes[sc_topk_idx]
            sc = sc[sc_topk_idx]

        keep = nms(boxes, sc, iou_thresh)
        boxes = boxes[keep]
        sc = sc[keep]
        results.append({'boxes': boxes, 'scores': sc})
    return results
