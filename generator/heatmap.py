import os
import cv2
import numpy as np
import random
# from typing import Union, Tuple, List
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from scipy.stats import skewnorm


def generate_skewed_blob(image_size, center, sigma_range, alpha=(-2.5,2.5), normalization=True):
    """
    Generate a skewed 2D blob on an empty canvas and center its centroid.

    :param image_size: Tuple[int, int] (height, width) of the output canvas.
    :param center: Tuple[int, int] (x, y) pixel coordinates where the blob centroid should end up.
    :param sigma_range: Tuple[float, float] range from which to sample the Gaussian standard deviations.
    :param alpha: Tuple[float, float] skewness parameter ranges (a_x, a_y) for the x and y axes.
    :param normalization: Boolean flag, if True normalize the blob peak value to 1.
    :return: 
        - canvas_shifted: np.ndarray of shape (height, width) with the skewed blob,
          normalized to [0, 1] if normalization is True and centered at `center`.
        - (sigma_x, sigma_y): Tuple[float, float] of the sampled standard deviations
          used along the x and y dimensions.
    """
    h, w = image_size
    cx, cy = center

    # sample parameters (keep as in your original function or make explicit)
    sigma_x = random.uniform(*sigma_range) * 1.05
    sigma_y = random.uniform(*sigma_range)
    alpha_x = random.uniform(*alpha)
    alpha_y = random.uniform(*alpha) * 0.95

    xs = np.arange(w) - cx
    ys = np.arange(h) - cy
    xv, yv = np.meshgrid(xs, ys)

    nx = xv / sigma_x
    ny = yv / sigma_y

    fx = skewnorm.pdf(nx, a=alpha_x)
    fy = skewnorm.pdf(ny, a=alpha_y)

    canvas = fx * fy
    if canvas.max() > 0 and normalization:
        canvas /= canvas.max()

    # compute centroid in pixel coordinates (weighted by canvas)
    total = canvas.sum()
    if total > 0:
        ys_idx = np.arange(h)[:, None]
        xs_idx = np.arange(w)[None, :]
        cx_blob = (canvas * xs_idx).sum() / total
        cy_blob = (canvas * ys_idx).sum() / total
        # current centroid coordinates are relative to absolute image indices, but because
        # we built xs,ys relative to cx,cy, the centroid is already in absolute pixel coords:
        # cx_blob, cy_blob are absolute x and y indices
    else:
        cx_blob, cy_blob = cx, cy

    # compute shift needed to move centroid to requested center
    shift_x = int(round(cx - cx_blob))
    shift_y = int(round(cy - cy_blob))

    # shift the canvas back so its centroid is at (cx, cy)
    canvas_shifted = np.zeros_like(canvas)
    # use roll then zero out wrapped parts for simplicity
    canvas_shifted = np.roll(canvas, shift=(shift_y, shift_x), axis=(0, 1))
    # zero-out wrapped stripes introduced by roll
    if shift_y > 0:
        canvas_shifted[:shift_y, :] = 0
    elif shift_y < 0:
        canvas_shifted[shift_y:, :] = 0
    if shift_x > 0:
        canvas_shifted[:, :shift_x] = 0
    elif shift_x < 0:
        canvas_shifted[:, shift_x:] = 0

    return canvas_shifted, (sigma_x, sigma_y)


def generate_gaussian_blob(image_size, center, sigma, normalization=True):
    """
    Create a single-channel Gaussian blob on an empty canvas.

    :param image_size: Tuple[int, int] (height, width) of the output image.
    :param center: Tuple[int, int] (x, y) pixel coordinates of the blob center.
    :param sigma: Float, standard deviation of the Gaussian.
    :return: np.ndarray of shape (height, width), values normalized to [0,1].
    """
    h, w = image_size
    canvas = np.zeros((h, w), dtype=np.float32)
    x, y = center

    if 0 <= x < w and 0 <= y < h:
        canvas[y, x] = 1

    sigma = random.randint(*sigma)
    canvas = gaussian_filter(canvas, sigma=sigma)
    
    if canvas.max() > 0 and normalization:
        canvas /= canvas.max()

    canvas *= norm.pdf(random.gauss(mu=0.0, sigma=1.0), loc=0, scale=1)*np.sqrt(2*np.pi)

    sigma = (sigma, sigma)

    return canvas, sigma


def generate_heatmap_and_boxes(image_size, num_blobs, sigma_range, offset_range, box_sizes):
    """
    Generate a heatmap with up to num_blobs Gaussian blobs clustered together,
    and compute each blob’s bounding box.

    :param image_size: Tuple[int, int] (height, width) of the image.
    :param num_blobs: Int, how many Gaussians to place (1 to 3).
    :param sigma_range: Float, Gaussian blur sigma for each blob.
    :param offset_range: Tuple[int, int], min and max pixel offset from the cluster center.
    :return:
        heatmap: np.ndarray of shape (h, w) with values in [0,1],
        boxes: List[Tuple[int, int, int, int]] as (x_min, y_min, x_max, y_max).
    """
    h, w = image_size

    if len(box_sizes) > 2:
       raise ValueError("Invalid input: len(box_sizes) must be at least 2.")

    margin = int(2*min(offset_range) + 5*max(sigma_range))
    cx = random.randint(margin, w - margin - 1)
    cy = random.randint(margin, h - margin - 1)

    centers = []
    for _ in range(0, num_blobs):
        angle = random.uniform(0, 2 * np.pi)
        dist = random.randint(offset_range[0], offset_range[1])
        # dx = int(dist * np.cos(angle))
        dx = float('inf')
        while cx + dx < 0 or cx + dx > h:
            dx = int(dist * np.cos(angle))

        dy = 0#int(dist * np.sin(angle))
        centers.append((cx + dx, cy + dy))

    heatmap = np.zeros((h, w), dtype=np.float32)
    boxes = []
    for x, y in centers:
        blob, sigma = generate_skewed_blob(image_size, (x, y), sigma_range)
        # blob, sigma = generate_gaussian_blob(image_size, (x, y), sigma)
        heatmap = np.maximum(heatmap, blob)

        xs = []
        for s in box_sizes:
            x_size = int(3*sigma[0] * s)  # roughly ±3σ
            y_size = int(3*sigma[1] * s)  # roughly ±3σ

            x_min = max(0, x - x_size // 2)
            y_min = max(0, y - y_size // 2)
            x_max = min(w - 1, x + x_size // 2)
            y_max = min(h - 1, y + y_size // 2)

            xs.append([x_min, y_min, x_max, y_max])
            
        boxes.append(tuple([x for xi in xs for x in xi]))

    return heatmap, boxes


def generate_dataset(num_images, image_size, sigma_range, offset_range, output_dir):
    """
    Create a dataset of synthetic **colored** heatmap images and box annotations.

    :param num_images: Int, total images to generate.
    :param image_size: Tuple[int, int], (height, width) of each image.
    :param sigma: Float, Gaussian blur sigma for blobs.
    :param offset_range: Tuple[int, int], min/max blob‐cluster offset.
    :param output_dir: Path where `images/` and `labels/` folders will be created.
    :return: None
    """
    img_dir = os.path.join(output_dir, 'images')
    lbl_dir = os.path.join(output_dir, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for idx in range(num_images):
        n_blobs = random.randint(1, MAX_BLOBS)
        heatmap, boxes = generate_heatmap_and_boxes(image_size, n_blobs, sigma_range, offset_range, box_sizes=BOX_SIZES)

        # ONE-BOX for all dist
        # boxes = np.array(boxes)
        # boxes = [tuple([boxes[:,0].min(), boxes[:,1].min(), boxes[:,2].max(), boxes[:,3].max(), boxes[:,4].min(), boxes[:,5].min(), boxes[:,6].max(), boxes[:,7].max()])]
        # END ONE-BOX for all dist
        
        # 1) Convert to 8-bit grayscale
        gray = (heatmap * 255).astype(np.uint8)

        # 2) Apply JET colormap → BGR color image
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        # 3) Save the colored heatmap
        img_path = os.path.join(img_dir, f'img_{idx:05d}.jpg')
        cv2.imwrite(img_path, colored)

        # 4) Save box coordinates
        lbl_path = os.path.join(lbl_dir, f'img_{idx:05d}.txt')
        with open(lbl_path, 'w') as f:
            for l_x_min, l_y_min, l_x_max, l_y_max, s_x_min, s_y_min, s_x_max, s_y_max in boxes:
                f.write(f"{l_x_min} {l_y_min} {l_x_max} {l_y_max} {s_x_min} {s_y_min} {s_x_max} {s_y_max}\n")

