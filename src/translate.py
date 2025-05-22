from transformers import MarianTokenizer, MarianMTModel

def translate(text):
    model_name= "Helsinki-NLP/opus-mt-ne-en"
    tokenizer= MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    output = tokenizer.decode(translated[0], skip_special_tokens = True)

    return output

if __name__ == "__main__":
    nepali_text = input("Enter Nepali text: ")
    english_text = translate(nepali_text)
    print("Translation: ", english_text)