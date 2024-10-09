from torch.utils.data import DataLoader
from models import FashionClipModel  # Importe a classe FashionClipModel do seu módulo
from dataset import FashionDataset
import torch
import os
import pickle

output_file='embeddings_bal.pkl'

model = FashionClipModel(model_name="patrickjohncyh/fashion-clip")

text_labels = [
  "shirt", "t-shirt", "hat", "shoes", "pants", "jeans", "dress", "skirt", "jacket", 
  "sweater", "coat", "shorts", "socks", "scarf", "gloves", "boots", "sandals", "suit", 
  "tie", "blazer", "blouse", "cardigan", "cotton", "linen", "wool", "silk", "denim", 
  "leather", "polyester", "nylon", "rayon", "spandex", "velvet", "satin", "chiffon", 
  "lace", "suede", "men", "women"
]

image_dir = '../roupas/images2'
dataset = FashionDataset(image_dir=image_dir, processor=model.processor, text_labels=text_labels)

# Configurar o DataLoader
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

embeddings = {}
i=0
# Iterar sobre o DataLoader para processar os lotes
for batch, image_paths in data_loader:
  batch = {key: value.to(model.device) for key, value in batch.items()}
  images = batch['pixel_values']
  images = images.reshape(images.shape[0], images.shape[2], images.shape[3], images.shape[4])
  
  with torch.no_grad():
    image_features = model.model.get_image_features(images).to('cpu')
  
  for img_features, img_path in zip(image_features, image_paths):
    file_name = os.path.basename(img_path)
    embeddings[file_name] = img_features
    print(f"Processed {i}/{len(dataset)}: {file_name}")
    i+=1
    
with open(output_file, 'wb') as f:
  pickle.dump(embeddings, f)
