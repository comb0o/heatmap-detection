import torch
import torch.nn as nn


class HeatmapDetector(nn.Module):
    def __init__(self, image_size, stride=None, scales=None, iou_threshold=0.25):
        super().__init__()
        self.image_size = image_size
        self.iou_threshold = iou_threshold

        self.conv1 = ConvBlock(3,  16, dropout=0.02)
        self.conv2 = ConvBlock(16, 32, dropout=0.05)
        self.conv3 = ConvBlock(32, 64, dropout=0.10)
        self.pool = nn.MaxPool2d(2, 2)

        if stride:
            self.stride = stride

        else:
            self.stride = self._infer_stride(torch.zeros(1, 3, *image_size))
        
        if scales.any():
            self.scales = scales

        else:
            self.scales = [self.stride*4, self.stride*6, self.stride*8, self.stride*10]

            
        ag = AnchorGenerator(stride=self.stride, scales=self.scales)
        num_anchors = len(ag.scales)
        
        self.register_buffer("anchors", ag.generate(image_size))
        # num_anchors = self.anchors.size(0) // ((image_size[0] // self.stride) * (image_size[1] // self.stride))
        self.detector = GridAnchorDetector(in_channels=64, num_anchors=num_anchors)

    def _infer_stride(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = self.conv3(x)
            feature_map_size = x.shape[2:]  # (H, W)

        stride_h = self.image_size[0] // feature_map_size[0]
        stride_w = self.image_size[1] // feature_map_size[1]

        assert stride_h == stride_w, "Non-square stride detected"
        return stride_h

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        
        offsets, logits = self.detector(x)
        
        return offsets, logits


class GridAnchorDetector(nn.Module):
    def __init__(self, in_channels=64, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.box_head = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        self.cls_head = nn.Conv2d(in_channels, num_anchors, kernel_size=1)

    def forward(self, feat):
        B, C, H, W = feat.shape
        # box_head: [B, num_anchors*4, H, W] -> reshape to [B, H, W, num_anchors, 4] -> [B, N, 4]
        box = self.box_head(feat)
        box = box.view(B, self.num_anchors, 4, H, W).permute(0, 3, 4, 1, 2).contiguous()
        offsets = box.view(B, H * W * self.num_anchors, 4)

        cls = self.cls_head(feat)
        cls = cls.view(B, self.num_anchors, 1, H, W).squeeze(2).permute(0, 2, 3, 1).contiguous()
        logits = cls.view(B, H * W * self.num_anchors)

        return offsets, logits


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation=nn.LeakyReLU, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            activation(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.block(x)


class AnchorGenerator():
    def __init__(self, stride=8, scales=[32, 64]):
        self.stride = stride
        self.scales = scales

    def generate(self, image_size):
        H = image_size[0] // self.stride
        W = image_size[1] // self.stride

        anchors = torch.empty((H * W * len(self.scales), 4), dtype=torch.float32)
        idx = 0
        for i in range(H):
            cy = (i + 0.5) * self.stride
            for j in range(W):
                cx = (j + 0.5) * self.stride
                
                for scale in self.scales:
                    scale = float(scale)
                    anchors[idx, 0] = cx - scale / 2
                    anchors[idx, 1] = cy - scale / 2
                    anchors[idx, 2] = cx + scale / 2
                    anchors[idx, 3] = cy + scale / 2
                    idx += 1

        return anchors



# __all__ = ["GridAnchorDetector","ConvBlock","AnchorGenerator","RoundObjectDetector"]