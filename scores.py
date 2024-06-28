import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Example code to calculate perplexity
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Encode input text
inputs = tokenizer.encode("input text here", return_tensors='pt')

# Generate output
with torch.no_grad():
    outputs = model(inputs, labels=inputs)
    loss = outputs.loss
    perplexity = torch.exp(loss)

print(f"Perplexity: {perplexity.item()}")

from rouge_score import rouge_scorer

# Example reference and candidate texts
references = ['reference sentence 1', 'reference sentence 2']
candidate = 'generated sentence'

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Calculate ROUGE scores
scores = scorer.score(' '.join(references), candidate)
print(f"ROUGE-1: {scores['rouge1'].fmeasure}")
print(f"ROUGE-2: {scores['rouge2'].fmeasure}")
print(f"ROUGE-L: {scores['rougeL'].fmeasure}")

from nltk.translate.bleu_score import corpus_bleu

# Example reference and candidate texts
references = [['reference sentence 1', 'reference sentence 2']]
candidates = ['generated sentence']

# Calculate BLEU score
bleu_score = corpus_bleu(references, candidates)
print(f"BLEU Score: {bleu_score}")

