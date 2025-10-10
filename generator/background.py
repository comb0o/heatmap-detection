import random
import torch
from io import BytesIO
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt


class CaptureWebsite():
    def __init__(self, url, image_size, headless=True):
        self.url = url
        self.width = image_size[0]
        self.height = image_size[1]
        self.tensors = []
        self._driver(url, headless)

    def _driver(self, url, headless):
        opts = Options()
        if headless:
            opts.add_argument("--headless")
            opts.add_argument(f"--window-size=1920,1080")  # initial size

        self.driver = webdriver.Chrome(options=opts)
        self.driver.get(url)
    
    def _crop(self):
        scroll_w = self.driver.execute_script("return document.body.scrollWidth")
        scroll_h = self.driver.execute_script("return document.body.scrollHeight")

        self.driver.set_window_size(scroll_w, scroll_h)
        png = self.driver.get_screenshot_as_png()

        full = Image.open(BytesIO(png)).convert("RGB")
        w, h = full.size

        max_x = max(0, w - self.width)
        max_y = max(0, h - self.height)
        x0 = random.randint(0, max_x)
        y0 = random.randint(0, max_y)
        
        crop = full.crop((x0, y0, x0 + self.width, y0 + self.height))
        crop = to_tensor(crop)

        return crop

    def _show_tensor(self, tensor, title=None):
        img = tensor.detach().cpu().clamp(0.0, 1.0)
        img = img.permute(1, 2, 0).numpy()

        plt.figure(figsize=(6, 6 * img.shape[0] / img.shape[1]))
        plt.imshow(img)
        plt.axis('off')
        if title:
            plt.title(title)
        
        plt.show()

    def run(self, num_images, show_images=False):
        for _ in range(num_images):
            tensor = self._crop()
            if show_images:
                print(type(tensor))
                self._show_tensor(tensor)

            self.tensors.append(tensor)
        
        self.close()

    def save(self, filename):
        torch.save(self.tensors, filename)

    def load(self, filename):
        self.tensors = torch.load(filename)

        return self.tensors

    def close(self):
        self.driver.quit()

