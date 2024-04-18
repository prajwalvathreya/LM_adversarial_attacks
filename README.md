# Adversarial Attacks On Language Models

This repository contains code for training a GPT-2-based transformer model for sentiment classification and generating adversarial examples. It utilizes FastAPI for creating a web API for testing.

## Files Overview

### `app.py`

This file defines a FastAPI application with endpoints for text classification inference and generating adversarial examples.

### `attacker_model_inference.py`

### load_target_model()

Loads the target sequence classification model trained on the IMDb dataset.

### load_attacker_model()

Loads the GPT-2 language model used as the attacker model.

### process_dataset(tokenizer, text, max_length)

Preprocesses the input text using the provided tokenizer and returns the processed dataset.

- `tokenizer`: The tokenizer object used for tokenization.
- `text`: The input text to be preprocessed.
- `max_length`: The maximum length of the input text.

### log_perplexity(logits, coeffs)

Calculates the log perplexity of the logits.

- `logits`: The logits produced by the model.
- `coeffs`: The coefficients for calculating log perplexity.

### calculate_word_error_rate(reference, hypothesis)

Calculates the Word Error Rate (WER) between reference and hypothesis sequences.

- `reference`: The reference sequence.
- `hypothesis`: The hypothesis sequence.

### load_embeddings(tokenizer, device, target_model, attacker_model)

Loads embeddings for both the target and attacker models.

- `tokenizer`: The tokenizer object used for tokenization.
- `device`: The device (e.g., 'cuda' or 'cpu') on which the embeddings will be loaded.
- `target_model`: The pre-trained target model.
- `attacker_model`: The pre-trained attacker model.

### train(target_model, attacker_model, input_ids, label, am_embeddings, tm_embeddings, device, tokenizer)

Trains the attacker model to generate adversarial examples.

- `target_model`: The pre-trained target model.
- `attacker_model`: The pre-trained attacker model.
- `input_ids`: The input IDs of the text sequence.
- `label`: The label of the input text sequence.
- `am_embeddings`: The embeddings of the attacker model.
- `tm_embeddings`: The embeddings of the target model.
- `device`: The device (e.g., 'cuda' or 'cpu') on which the training will be performed.
- `tokenizer`: The tokenizer object used for tokenization.


### `attacker_model.py` - Training/ experimentation script

### load_target_model()

Loads the target sequence classification model trained on the IMDb dataset.

### load_attacker_model()

Loads the GPT-2 language model used as the attacker model.

### process_dataset(dataset, max_length=256)

Processes the dataset for model input using the GPT-2 tokenizer.

- `dataset`: The dataset to be processed.
- `max_length`: The maximum length of the input sequences (default is 256).

### log_perplexity(logits, coeffs)

Calculates the log perplexity of the logits.

- `logits`: The logits produced by the model.
- `coeffs`: The coefficients for calculating log perplexity.

### calculate_word_error_rate(reference, hypothesis)

Calculates the Word Error Rate (WER) between reference and hypothesis sequences.

- `reference`: The reference sequence.
- `hypothesis`: The hypothesis sequence.

### load_embeddings()

Loads embeddings for both the target and attacker models.

### train(target_model, attacker_model, inputs, labels, am_embeddings, tm_embeddings)

Trains the attacker model to generate adversarial examples.

- `target_model`: The pre-trained target model.
- `attacker_model`: The pre-trained attacker model.
- `inputs`: The input sequences.
- `labels`: The corresponding labels for the input sequences.
- `am_embeddings`: The embeddings of the attacker model.
- `tm_embeddings`: The embeddings of the target model.

### `classification_inference.py`

### PositionalEncoding

Implements the positional encoding for transformer models.

#### Parameters

- `d_model`: The dimensionality of the model.
- `max_len`: The maximum length of the input sequence (default is 5000).

#### Methods

- `__init__(self, d_model, max_len=5000)`: Initializes the PositionalEncoding module.
- `forward(self, x)`: Adds positional encoding to the input tensor.

### TransformerDecoder

Implements the decoder part of a transformer model.

#### Parameters

- `vocab_size`: The size of the vocabulary.
- `embed_dim`: The dimensionality of the embeddings (default is 32).
- `num_heads`: The number of attention heads (default is 2).
- `hidden_dim`: The dimensionality of the hidden layer (default is 128).
- `num_layers`: The number of decoder layers (default is 1).
- `dropout`: The dropout probability (default is 0.1).

#### Methods

- `__init__(self, vocab_size, embed_dim=32, num_heads=2, hidden_dim=128, num_layers=1, dropout=0.1)`: Initializes the TransformerDecoder module.
- `init_weights(self)`: Initializes the weights of the model.
- `forward(self, input_ids)`: Performs forward pass through the decoder.

### init_model

Initializes the transformer decoder model.

#### Parameters

- `tokenizer`: The tokenizer object.
- `device`: The device to use for computation.

#### Returns

- `model`: The initialized transformer decoder model.

### inference

Performs inference using the transformer decoder model.

#### Parameters

- `model`: The transformer decoder model.
- `tokenizer`: The tokenizer object.
- `device`: The device to use for computation.
- `text`: The input text for inference.

#### Returns

- `output`: The output probability.


### `classification_model.py` - Training/ experimentation script

### IMDbDataset

Custom dataset class for loading the IMDb dataset.

#### Parameters

- `dataset`: The dataset object loaded using `load_dataset`.
- `tokenizer`: The GPT2 tokenizer object.
- `max_length`: The maximum length of input sequences (default is 384).

#### Methods

- `__init__(self, dataset, tokenizer, max_length=384)`: Initializes the IMDbDataset class.
- `__len__(self)`: Returns the length of the dataset.
- `__getitem__(self, idx)`: Returns a single sample from the dataset.

### PositionalEncoding

Implements the positional encoding for transformer models.

#### Parameters

- `d_model`: The dimensionality of the model.
- `max_len`: The maximum length of the input sequence (default is 5000).

#### Methods

- `__init__(self, d_model, max_len=5000)`: Initializes the PositionalEncoding module.
- `forward(self, x)`: Adds positional encoding to the input tensor.

### TransformerDecoder

Implements the decoder part of a transformer model.

#### Parameters

- `vocab_size`: The size of the vocabulary.
- `embed_dim`: The dimensionality of the embeddings (default is 768).
- `num_heads`: The number of attention heads (default is 2).
- `hidden_dim`: The dimensionality of the hidden layer (default is 2048).
- `num_layers`: The number of decoder layers (default is 2).
- `dropout`: The dropout probability (default is 0.1).

#### Methods

- `__init__(self, vocab_size, embed_dim=768, num_heads=2, hidden_dim=2048, num_layers=2, dropout=0.1)`: Initializes the TransformerDecoder module.
- `init_weights(self)`: Initializes the weights of the model.
- `forward(self, input_ids, input_embeds=None)`: Performs forward pass through the decoder.

### train

Function to train the TransformerDecoder model.

#### Parameters

- `model`: The TransformerDecoder model.
- `data_loader`: The DataLoader object for loading data.
- `epochs`: The number of training epochs (default is 40).

### main

Entry point of the script.

#### Calls

- `train(model, data_loader)`: Trains the TransformerDecoder model using the provided DataLoader.



## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/prajwalvathreya/LM_adversarial_attacks.git

2. **Install Dependencies:**
    
    ```bash
    pip install -r requirements.txt

3. **Start FastAPI Server:**

    ```bash
    uvicorn app:app --reload

4. **Use /docs for testing the API:**

    ```bash
    http://YOUR_PORT_URL/docs

## References

- Author(s): Chuan Gu, Alexandre Sablayrolles, Hervé Jégou, Douwe Kiela  : Facebook AI Research
- Title: "Gradient-based Adversarial Attacks against Text Transformers"
- [Link to Paper](https://arxiv.org/pdf/2104.13733.pdf)

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Transformers Documentation](https://huggingface.co/transformers/)
- [IMDB Dataset](https://huggingface.co/datasets/imdb)