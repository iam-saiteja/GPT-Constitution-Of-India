# GPT-Constitution-Of-India
# This is demo and trained on Non Verified Dataset and may give inappropriate outputs based on prompts and the model may acts as biased because the model is not trained heavily because of the system hardware limitation

This repository contains scripts for fine-tuning a GPT-2 model on Indian legal and constitutional texts and generating text responses based on user prompts.

## Overview

This project fine-tunes a GPT-2 model on the text of the Indian Constitution. The fine-tuned model can then be used to generate meaningful text responses to various prompts related to Indian legal and constitutional topics.

## Repository Structure

- `Training.py`: To fine-tuning the GPT-2 model.
- `Working.py`: To To generating text using the fine-tuned model, generating text based on user input with improved prompt handling.
- `combined_text.txt`: Combined text file of the Indian Constitution and index documents used for training.
- `fine_tuned_model/`: Directory containing the fine-tuned GPT-2 model and tokenizer.

## Setup

### Prerequisites

- Python 3.7+
- `transformers` library
- `datasets` library
- `pandas` library

## Usage

### Fine-Tuning

To fine-tune the GPT-2 model, run the `Training.py` script. The execution of this file will automatically generate the combined_text.txt file

### Generating Text

To generate text using the fine-tuned model, run the `Working.py` script. You will be prompted to enter a prompt, and the model will generate a response.

### Example

```sh
C:\Users\iamsai\Documents\Postulate\Demo\archive> python Working.py
Enter Prompt: How does the Indian Constitution protect the rights of minorities?
```

Output:
```
[{'generated_text': 'How does the Indian Constitution protect the rights of minorities? It is the duty of the Government of India, under a Constitution which guarantees fundamental rights and personal liberties, to make it possible for the peoples to exercise the right of selfgovernment in all public and private affairs for free and unimpaired enjoyment of the same, and in that light is represented by its protection of civil liberty and the right from abdication to citizenship, in so far as the rights conferred by such Constitution are guaranteed by law and the laws, if the Government so wishes, are in force and have the force and power conferred by such Constitution. The Constitution of India guarantees that the rights conferred by law, or by the laws, guaranteed by the Constitution are the guarantees conferred by the States aforesaid on the State at its disposal, and that it shall not be abdicated to the people to exercise the right of selfgovernment in all public and private affairs for free and unimpaired enjoyment of the same\n49
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss improvements or fixes.

## License

This project is licensed under the MIT License.
