import numpy as np
import pickle
import os
import faiss
import matplotlib
matplotlib.use('TkAgg')  # Define um backend interativo
import matplotlib.pyplot as plt
from PIL import Image


embeddings_path = '/home/baldoqui/ssd/roupas/embeddings_bal.pkl'

# Carregar os embeddings usando numpy.load() com allow_pickle=True
with open(embeddings_path, 'rb') as f:
    loaded_embeddings = pickle.load(f)

print("Tipo do conteúdo:", type(loaded_embeddings))

image_dir = '../roupas/images2/'

embedding_values = np.array(list(loaded_embeddings.values()))
embedding_values = embedding_values.reshape(embedding_values.shape[0],768)

image_files = list(loaded_embeddings.keys())


d = embedding_values.shape[1]
print(d)
index = faiss.IndexFlatL2(d)  # Indexador de similaridade com L2 (euclidiana)
index.add(embedding_values)  # Adicionar os embeddings ao indexador

def calcular_distancia(a, b):
    return np.linalg.norm(a - b)



def search_similar_images(query_embedding, k=5, min_distance=0):
    filtered_indices = []
    results_to_fetch = k + 100
    
    while len(filtered_indices) < k:
        distances, indices = index.search(np.array([query_embedding]), results_to_fetch)

        for dist, i in zip(distances[0], indices[0]):
            # Obtenha o embedding da nova imagem
            new_embedding = embedding_values[i]

            # Verifique se a nova imagem é distante de todas as já filtradas
            is_distante = True
            for idx in filtered_indices:
                existing_embedding = embedding_values[idx]
                if calcular_distancia(new_embedding, existing_embedding) < min_distance:
                    is_distante = False
                    break

            if is_distante:
                filtered_indices.append(i)

    similar_images = [image_files[i] for i in filtered_indices[:k]]

    return similar_images



def display_similar_images(similar_images):
    plt.figure(figsize=(15, 5))

    for i, image_file in enumerate(similar_images):
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)

        # Exibe a imagem em uma subtrama
        ax = plt.subplot(1, len(similar_images), i + 1)
        plt.imshow(image)
        plt.axis('off')  # Oculta os eixos
        ax.set_title(f"Imagem {i + 1}")

    plt.show()

# Buscar as imagens mais similares e exibi-las
similar_images = search_similar_images(embedding_values[1], k=5)
print(similar_images)
display_similar_images(similar_images)
