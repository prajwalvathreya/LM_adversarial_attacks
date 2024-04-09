from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2Tokenizer
from classification_inference import inference, init_model
from attacker_model_inference import *
import torch


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify a list of allowed origins or use "*" to allow all
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

inf_model = init_model(tokenizer, device)
inf_model.eval()

class InputData(BaseModel):
    input_text: str


@app.get("/")
def home():
    return {}

@app.post("/inference")
def run_inference(input_data: InputData):
    
    # Perform inference on the input_text
    result = inference(inf_model, tokenizer, device, input_data.input_text)

    return {"result": result}


@app.post("/adversarial_example")
def gen_adversarial_example(input_data: InputData, input_label: int):

    target_model = load_target_model().to(device)
    attacker_model = load_attacker_model().to(device)
    processed_dataset = process_dataset(tokenizer, input_data.input_text)
    tm_embeddings, am_embeddings = load_embeddings(tokenizer, device, target_model, attacker_model)
    original_pred, original_text, adversarial_text, adversarial_pred = train(target_model, attacker_model, processed_dataset['input_ids'], label = torch.tensor(input_label, dtype=torch.long), am_embeddings = am_embeddings.to(device), tm_embeddings = tm_embeddings.to(device), device = device, tokenizer = tokenizer)
    
    return {"original_text": original_text, "original_pred": original_pred.item(), "adversarial_text": adversarial_text, "adversarial_pred": adversarial_pred.item()}