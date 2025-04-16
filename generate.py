from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "./gpt2-finetuned"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

def generate_text(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.5,
        do_sample=True,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
