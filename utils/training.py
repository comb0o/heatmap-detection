import torch
import torch.nn.functional as F
from utils.geometry import assign_targets_to_anchors, encode_boxes_to_offsets

def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    # logits: [B, N], targets: [B, N] in {0,1}
    prob = torch.sigmoid(logits)

    # cross-entropy per element
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    # p_t = prob if target==1 else (1-prob)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    
    # focal factor
    focal_factor = (1 - p_t).pow(gamma)
    
    # alpha weighting
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_factor * focal_factor * ce

    if reduction == 'sum':
        return loss.sum()
    
    elif reduction == 'none':
        return loss
    
    return loss.mean()


class Experiment():
    def __init__(self, model, device, optimizer, scheduler, max_grad_norm=1, warmup_epochs=5, writer=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.warmup_epochs = warmup_epochs
        self.writer = writer
        self.global_step = 0

    def train_one_epoch(self, dataloader, epoch, lambda_cls, lambda_reg, focal_gamma):
        self.model.train()
        running_loss = 0.0
        anchors = self.model.anchors.to(self.device)
        
        use_focal = epoch > self.warmup_epochs
        for images, gt_boxes in dataloader:
            images = images.to(self.device)           # [B, 3, H, W]
            gt_boxes = gt_boxes['s_boxes'].to(self.device)         # [B, M, 4] padded with zeros
    
            # forward
            pred_offsets, pred_logits = self.model(images)   # [B, N, 4], [B, N]
    
            # assign anchors → targets
            target_boxes, object_mask = assign_targets_to_anchors(gt_boxes, anchors, iou_threshold=self.model.iou_threshold)
    
            # encode boxes
            target_offsets = encode_boxes_to_offsets(target_boxes, anchors)
    
            # classification loss
            if use_focal:
                cls_loss = focal_loss(pred_logits, object_mask, gamma=focal_gamma, reduction='sum')
                
            else:
                cls_loss = F.binary_cross_entropy_with_logits(pred_logits, object_mask, reduction='sum')
    
            cls_loss = cls_loss / object_mask.numel()
    
            pos_mask = object_mask.bool()           # [B, N]
            if pos_mask.any():
                reg_loss = F.smooth_l1_loss(pred_offsets[pos_mask], target_offsets[pos_mask], reduction='sum')
                reg_loss = reg_loss / pos_mask.sum()
                
            else:
                reg_loss = torch.tensor(0.0, device=self.device)
    
            # total loss
            loss = lambda_cls * cls_loss + lambda_reg * reg_loss
    
            # backward & optimize
            self.optimizer.zero_grad()
            loss.backward()
    
            # 2. Log gradient norms per layer
            if self.writer is not None:
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        grad_norm = p.grad.norm().item()
                        self.writer.add_scalar(f"grad_norm/{name}", grad_norm, self.global_step)
                
                # 3. Log current learning rate per param group
                for i, pg in enumerate(self.optimizer.param_groups):
                    lr = pg["lr"]
                    self.writer.add_scalar(f"learning_rate/group_{i}", lr, self.global_step)
            
            # for name, p in model.named_parameters():
            #     if p.grad is not None:
            #         mean_grad = p.grad.abs().mean().item()
            #         mean_weight = p.data.abs().mean().item()
            #         ratio = mean_grad / (mean_weight + 1e-8)
    
            #         print(name, ratio)
    
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
    
            self.global_step += 1
    
            # 4. At epoch end: log weight distributions
            if self.writer is not None:
                for name, p in self.model.named_parameters():
                    self.writer.add_histogram(f"weights/{name}", p.data.cpu().numpy(), epoch)
    
            # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # print(f"Grad norm: {total_norm:.2f}")
    
            running_loss += loss.item() * images.size(0)
    
        epoch_loss = running_loss / len(dataloader.dataset)
        
        return epoch_loss
    
    
    def validate_one_epoch(self, dataloader, lambda_cls, lambda_reg):
        self.model.eval()
        running_loss = 0.0
        anchors = self.model.anchors.to(self.device)
    
        with torch.no_grad():
            for images, gt_boxes in dataloader:
                images = images.to(self.device)
                gt_boxes = gt_boxes['s_boxes'].to(self.device)
    
                pred_offsets, pred_logits = self.model(images)
                target_boxes, object_mask = assign_targets_to_anchors(gt_boxes, anchors, iou_threshold=self.model.iou_threshold)
                target_offsets = encode_boxes_to_offsets(target_boxes, anchors)
    
                cls_loss = F.binary_cross_entropy_with_logits(pred_logits, object_mask)
                pos_mask = object_mask.bool()
                if pos_mask.any():
                    reg_loss = F.smooth_l1_loss(pred_offsets[pos_mask], target_offsets[pos_mask], reduction='mean')
                    
                else:
                    reg_loss = torch.tensor(0.0, device=self.device)
    
                loss = lambda_cls * cls_loss + lambda_reg * reg_loss
                running_loss += loss.item() * images.size(0)
    
        return running_loss / len(dataloader.dataset)
