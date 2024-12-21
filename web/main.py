from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import os
from typing import Dict, Optional
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
from omegaconf import OmegaConf

from src.data.tokenizer import TranslationTokenizer
from src.utils.device import get_device

app = FastAPI()
app.mount("/static", StaticFiles(directory="web/static"), name="static")

models: Dict[str, torch.nn.Module] = {}
tokenizer: Optional[TranslationTokenizer] = None
web_config: Optional[dict] = None


class TranslationRequest(BaseModel):
    text: str
    model_id: str

def load_web_config():
    global web_config
    config_path = Path("web/config/web_config.yaml")
    web_config = OmegaConf.load(config_path)
    return web_config



def load_model_config(model_name: str):
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=web_config.base_ml_config):
        cfg = compose(
            config_name="config",
            overrides=[f"model={model_name}"]
        )
        return cfg

def load_models():
    global models, tokenizer, web_config
    
    device = get_device("auto")
    print(f"Using device: {device}")
    
    tokenizer = TranslationTokenizer(
        name="facebook/wmt19-ru-en",
        max_length=128
    )
    
    for model_id, model_cfg in web_config.models.items():
        # try:
            # Load model config using Hydra
            model_config = load_model_config(model_cfg.name)
            print(f"Loaded config for model {model_id}")
            print(model_config)
            
            model_config.model.vocab_size = tokenizer.vocab_size
            model_config.model.pad_token_id = tokenizer.pad_token_id
            
            model = hydra.utils.instantiate(model_config.model.seq2seq).to(device)
            
            checkpoint_dir = Path("checkpoints") / model_cfg.checkpoint_dir
            checkpoints = list(checkpoint_dir.glob("best_model*.pt"))
            if not checkpoints:
                print(f"No checkpoint found for model {model_id}")
                continue
            
            print(f"Loading checkpoint: {checkpoints[0]}")
            checkpoint = torch.load(checkpoints[0], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            models[model_id] = model
            print(f"Successfully loaded model: {model_id}")
            

@app.on_event("startup")
async def startup_event():
    load_web_config()
    load_models()

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("web/static/index.html") as f:
        return f.read()

@app.get("/models")
async def get_models():
    models_info = {
        "models": [
            {
                "id": model_id,
                "name": cfg.title,
                "description": cfg.description,
                "max_length": cfg.max_length
            }
            for model_id, cfg in web_config.models.items()
            if model_id in models
        ]
    }
    print(models_info)
    return models_info

@app.post("/translate")
async def translate(request: TranslationRequest):
    print("translate request")
    if request.model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_cfg = web_config.models[request.model_id]
    
    if len(request.text.split()) > model_cfg.max_length:
        raise HTTPException(
            status_code=400, 
            detail=f"Text too long. Maximum length for this model is {model_cfg.max_length} words"
        )
    
    inputs = tokenizer.encode(
        request.text,
        return_tensors="pt"
    )["input_ids"].to(next(models[request.model_id].parameters()).device)
    
    with torch.no_grad():
        token_indices = models[request.model_id].translate(inputs)
        translated_text = tokenizer.decode(token_indices[0], skip_special_tokens=True)
    
    return {"translation": translated_text}