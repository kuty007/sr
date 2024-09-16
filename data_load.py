from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        print(len(self.lr_images))
        print(len(self.hr_images))

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])

        # Open images and convert to RGB
        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')

        # Apply your transforms and return
        hr_image = self.transform(hr_image)
        lr_image = self.transform(lr_image)

        return lr_image, hr_image


# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])


