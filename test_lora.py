from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

model_name = "cowden_lora_fixed_model"

print(f"Loading {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
print("Loaded! Ask BabyRapa anything (type 'exit' to quit)\n")

while True:
    question = input("### Question: ")
    if question.lower() in ["exit", "quit", "q"]:
        break
    if not question.strip():
        continue

    prompt = f"""### Question:
{question}

### Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
   
    print("\n### Answer: ", end="")
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=300,
        use_cache=True,
        temperature=0.7,
        top_p=0.9,
    )
    print("\n" + "-"*60 + "\n")