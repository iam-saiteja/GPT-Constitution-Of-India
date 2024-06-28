from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

tuned_model = './fine_tuned_model'
tokenizer = GPT2Tokenizer.from_pretrained(tuned_model)
model = GPT2LMHeadModel.from_pretrained(tuned_model)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

prompt = input("Enter Prompt:- ")
output = generator(prompt, max_length=250, num_return_sequences=1, truncation=True)
print(output)
