import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

data_path_1 = 'Constitution Of India.csv'
data_path_2 = 'Index.csv'

df1 = pd.read_csv(data_path_1, encoding='ISO-8859-1')
df2 = pd.read_csv(data_path_2, encoding='ISO-8859-1')

combined_text = df1.to_string() + df2.to_string()

combined_text_path = 'combined_text.txt'
with open(combined_text_path, 'w', encoding='ISO-8859-1') as f:
    f.write(combined_text)

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

max_length = 1024 
chunks = [combined_text[i:i+max_length] for i in range(0, len(combined_text), max_length)]

chunks = chunks[:500]

dataset = Dataset.from_dict({'text': chunks})

def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True)

dataset = dataset.map(encode, batched=True, remove_columns=['text'])

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1, 
    per_device_train_batch_size=2, #may be three or four overcomming hardware limitation
    save_steps=10_000,
    save_total_limit=2,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
