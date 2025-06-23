from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch
import re

app = FastAPI()

# Load tokenizer and model
model_path = "./mt5-npi-en"
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()

# Request models
class TranslationInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: list[str]

# === Utility ===
def split_into_sentences_nepali(text):
    # Split on danda (।), question mark (?) or exclamation (!)
    sentences = re.split(r'[।?!|]', text)
    # Strip whitespace and remove empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def chunk_text(text, tokenizer, max_tokens=300):
    sentences = split_into_sentences_nepali(text)  # Use regex based splitting here
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        temp_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        tokenized = tokenizer("translate Nepali to English: " + temp_chunk, return_tensors="pt", truncation=False)
        
        if tokenized.input_ids.shape[1] > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = temp_chunk

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# === Translation ===
def translate_nepali_to_english(text: str) -> str:
    sentences = split_into_sentences_nepali(text)
    translated_sentences = []

    for sentence in sentences:
        chunks = chunk_text(sentence, tokenizer, max_tokens=300)
        translated_chunks = []

        for chunk in chunks:
            input_text = "translate Nepali to English: " + chunk
            encoding = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    max_length=512,
                    num_beams=4
                )

            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_chunks.append(translated)

        translated_sentences.append(" ".join(translated_chunks))

    return " . ".join(translated_sentences)


# === Endpoints ===
@app.post("/translate")
def translate(input: TranslationInput):
    translated = translate_nepali_to_english(input.text)
    return {"translated_text": translated}

@app.post("/translate_batch")
def translate_batch(input: BatchInput):
    results = [translate_nepali_to_english(t) for t in input.texts]
    return {"results": results}
