from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    MarianMTModel, MarianTokenizer
)
import torch

model_ckpt = "papluca/xlm-roberta-base-language-detection"
tokenizer_detected = AutoTokenizer.from_pretrained(model_ckpt)
model_detected = AutoModelForSequenceClassification.from_pretrained(model_ckpt)

# result_list = []
# inputs = tokenizer_detected(list_text, padding=True, truncation=True, return_tensors="pt")
# with torch.no_grad():
#     logits = model_detected(**inputs).logits

def detect_language(text):
    inputs = tokenizer_detected(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model_detected(**inputs).logits
    preds = torch.softmax(logits, dim=-1)
    id2lang = model_detected.config.id2label
    vals, idxs = torch.max(preds, dim=1)
    return [(id2lang[k.item()], v.item()) for k, v in zip(idxs, vals)]

# preds = torch.softmax(logits, dim=-1)
# id2lang = model_detected.config.id2label
# vals, idxs = torch.max(preds, dim=1)
# result_list.extend([(id2lang[k.item()], v.item()) for k, v in zip(idxs, vals)])

# result_list


model_name = "Helsinki-NLP/opus-mt-es-en"
# model_name = "Helsinki-NLP/opus-mt-en-es"
# model_name = "Helsinki-NLP/opus-mt-tc-big-en-es"

tokenizer_tradu = MarianTokenizer.from_pretrained(model_name)
model_tradu = MarianMTModel.from_pretrained(model_name)

def get_translation(text):
  translated = model_tradu.generate(**tokenizer_tradu(text, return_tensors="pt", padding=True))
  return tokenizer_tradu.decode(translated[0], skip_special_tokens=True)

