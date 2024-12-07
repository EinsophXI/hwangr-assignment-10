import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from open_clip import create_model_and_transforms, tokenizer
import open_clip
from PIL import Image
import os
import numpy as np

# Load the model and embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Load the original CLIP embeddings
df = pd.read_pickle('image_embeddings.pickle')

# Perform PCA on the CLIP embeddings
pca = PCA(n_components=512)  # Number of principal components to keep
clip_embeddings = [row['embedding'] for _, row in df.iterrows()]
pca_embeddings = pca.fit_transform(clip_embeddings)

# Create a new DataFrame with PCA embeddings and associated filenames
pca_embeddings_df = pd.DataFrame(pca_embeddings, columns=[f'PC{i+1}' for i in range(pca_embeddings.shape[1])])
pca_embeddings_df['file_name'] = df['file_name']

# Save the PCA embeddings to a pickle file for future use
pca_embeddings_df.to_pickle('pca_embeddings.pickle')



# Cosine similarity function
def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2, dim=1).item()

# Function to find best matches based on the selected embedding type (CLIP or PCA)
def _find_best_matches(query_embedding, top_k=5, use_pca=False):
    results = []

    query_embedding = query_embedding.to('cpu')  # Ensure embeddings are on the same device

    if use_pca:
        # Use PCA embeddings for comparison
        for _, row in pca_embeddings_df.iterrows():
            # Drop 'file_name' and ensure the remaining data is numeric
            embedding = torch.tensor(row.drop('file_name').to_numpy(dtype=np.float32), dtype=torch.float32)
            similarity = cosine_similarity(query_embedding, embedding)
            results.append((row['file_name'], similarity))
    else:
        # Use CLIP embeddings for comparison
        for _, row in df.iterrows():
            embedding = torch.tensor(row['embedding'], dtype=torch.float32)
            similarity = cosine_similarity(query_embedding, embedding)
            results.append((row['file_name'], similarity))

    # Sort by similarity in descending order and return the top `top_k` results
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    return results


# Perform image search using either CLIP or PCA embeddings
def perform_image_search(image_path, use_pca=False):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    query_embedding = F.normalize(model.encode_image(image))  # Normalize query embedding
    return _find_best_matches(query_embedding, use_pca=use_pca)

# Perform text search (no change needed for text search logic)
def perform_text_search(text_query, use_pca=False):
    text_tokenized = tokenizer([text_query])
    query_embedding = F.normalize(model.encode_text(text_tokenized))
    return _find_best_matches(query_embedding, use_pca=use_pca)

# Perform hybrid search (combining text and image queries with weighting)
def perform_hybrid_search(text_query, image_path, lam=0.5, use_pca=False):
    text_tokenized = tokenizer([text_query])
    text_embedding = F.normalize(model.encode_text(text_tokenized))

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    image_embedding = F.normalize(model.encode_image(image))

    query_embedding = F.normalize(lam * text_embedding + (1 - lam) * image_embedding)

    return _find_best_matches(query_embedding, use_pca=use_pca)
