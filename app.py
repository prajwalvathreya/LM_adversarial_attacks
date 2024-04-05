from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2Tokenizer
from transformers import GPT2Model
from classification_inference import inference, init_model
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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