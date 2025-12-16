from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np


model = SentenceTransformer('all-MiniLM-L6-v2')


def create_embeddings(csv_path, index_path):
df = pd.read_csv(csv_path)
embeddings = model.encode(df['chunk'].tolist(), show_progress_bar=True)


dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))


faiss.write_index(index, index_path)
return df
