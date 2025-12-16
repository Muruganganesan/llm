from transformers import pipeline


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def summarize(text, max_len):
return summarizer(text, max_length=max_len, min_length=50, do_sample=False)[0]['summary_text']
