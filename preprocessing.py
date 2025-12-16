import pandas as pd
import re


CHUNK_SIZE = 300


def clean_text(text):
text = re.sub(r"[^a-zA-Z0-9\.\, ]", "", text)
text = text.lower()
return text


def chunk_text(text, size=CHUNK_SIZE):
words = text.split()
return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


def preprocess(input_path, output_path):
df = pd.read_csv(input_path)
chunks = []


for doc in df['text']:
cleaned = clean_text(doc)
for chunk in chunk_text(cleaned):
chunks.append(chunk)


out_df = pd.DataFrame({'chunk': chunks})
out_df.to_csv(output_path, index=False)
