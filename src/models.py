import torch
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor

class FashionClipModel:
    def __init__(self, model_name="patrickjohncyh/fashion-clip"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def extract_embedding(self, input):

        with torch.no_grad():
            image_features = self.model.get_image_features(input['pixel_values']).to('cpu')
        
        return image_features.numpy()
    
class MarqoFashionClipModel:
    def __init__(self, model_name="Marqo/marqo-fashionSigLIP"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def extract_embedding(self, image_path):
        image = Image.open(image_path)
        text = [
            "shirt", "t-shirt", "hat", "shoes", "pants", "jeans", "dress", "skirt", "jacket", 
            "sweater", "coat", "shorts", "socks", "scarf", "gloves", "boots", "sandals", "suit", 
            "tie", "blazer", "blouse", "cardigan", "cotton", "linen", "wool", "silk", "denim", 
            "leather", "polyester", "nylon", "rayon", "spandex", "velvet", "satin", "chiffon", 
            "lace", "suede", "men", "women"
        ]
        
        processed = self.processor(text=text, images=image, padding=True, return_tensors="pt")
        processed = processed.to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(processed['pixel_values']).to('cpu')
        
        return image_features.numpy()