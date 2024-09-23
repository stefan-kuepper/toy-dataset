from typing import Tuple
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import random


class ToyDataset(Dataset):
    def __init__(
        self,
        num_images: int,
        canvas_size: Tuple[int, int],
        object_size,
        seed=23,
    ):
        super().__init__()
        random.seed(seed)
        self.num_images = num_images
        self.canvas_size = canvas_size
        self.object_size: int = object_size
        self.objects_per_image: int = 3
        self.images = []
        self.centers = []
        for img_num in range(self.num_images):
            img = Image.new(mode="L", size=self.canvas_size, color=0)
            d = ImageDraw.Draw(img)
            centers = []
            for obj in range(random.randrange(self.objects_per_image)):
                ox = random.randrange(
                    0 + self.object_size, self.canvas_size[0] - self.object_size
                )
                oy = random.randrange(
                    0 + self.object_size, self.canvas_size[1] - self.object_size
                )
                centers.append((ox, oy))

            for cx, cy in centers:
                d.rectangle(
                    (
                        cx - self.object_size / 2,
                        cy - self.object_size / 2,
                        cx + self.object_size / 2,
                        cy + self.object_size / 2,
                    ),
                    fill=127,
                    outline=255,
                )

            self.images.append(img)
            self.centers.append(centers)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index: int):
        return self.images[index], self.centers[index]
