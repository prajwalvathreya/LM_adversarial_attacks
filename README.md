# LM_adversarial_attacks_

This repository contains code for training a GPT-2-based transformer model for text classification and generating adversarial examples. It utilizes FastAPI for creating a web API, transformers for model loading and processing, and PyTorch for training and inference.

## Files Overview

### `app.py`

This file defines a FastAPI application with endpoints for text classification inference and generating adversarial examples.

### `attacker_model_inference.py`

Provides functions for loading target and attacker models, processing datasets, and generating adversarial examples.

### `attacker_model.py`

Defines functions for loading models, processing datasets, and training the attacker model to generate adversarial examples.

### `classification_inference.py`

Contains a transformer-based decoder model for text classification inference.

### `classification_model_cuda.py`

Defines a transformer-based decoder model for training a text classification model.

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
