import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2')


def search(query, index_path, documents, top_k=5):
index = faiss.read_index(index_path)
query_vec = model.encode([query])
distances, indices = index.search(np.array(query_vec), top_k)
return [documents[i] for i in indices[0]]
