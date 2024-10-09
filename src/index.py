import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from transformers import AutoModel, AutoProcessor
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo e o processador
model = AutoModel.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)

model.to(device)

def extract_embedding(image_path):
  image = Image.open(image_path)
  text = [
    "shirt", "t-shirt", "hat", "shoes", "pants", "jeans", "dress", "skirt", "jacket", 
    "sweater", "coat", "shorts", "socks", "scarf", "gloves", "boots", "sandals", "suit", 
    "tie", "blazer", "blouse", "cardigan", "cotton", "linen", "wool", "silk", "denim", 
    "leather", "polyester", "nylon", "rayon", "spandex", "velvet", "satin", "chiffon", 
    "lace", "suede", "men", "women"
  ]
  
  processed = processor(text=text, images=image, padding='max_length', return_tensors="pt")

  processed.to(device)

  with torch.no_grad():
    image_features = model.get_image_features(processed['pixel_values'], normalize=True).to('cpu')
     
  return image_features.numpy()

image_dir = '../roupas/images2'

# Listar todos os arquivos .jpg na pasta 'roupas'
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]

print(image_paths[1])

# Extrair os embeddings e armazená-los na lista 'embeddings'
embeddings = {}
i=0
for image_path in image_paths:
    emb = extract_embedding(image_path)
    
    file_name = os.path.basename(image_path)  # Obtém apenas o nome do arquivo

    print(i)
    i+=1
    embeddings[file_name] = emb
    # if(i>100):
    #   break

with open('embeddings_bal.pkl', 'wb') as f:
  pickle.dump(embeddings, f)


