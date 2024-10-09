import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class FashionDataset(Dataset):
    def __init__(self, image_dir, processor, text_labels):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]
        self.processor = processor
        self.text_labels = text_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        processed = self.processor(text=self.text_labels, images=image, padding=True, return_tensors="pt")
        return processed, image_path  # Retornamos também o caminho da imagem para referência
