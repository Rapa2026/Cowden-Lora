from unsloth import FastLanguageModel
import gradio as gr

model, tokenizer = FastLanguageModel.from_pretrained(
    "cowden_lora_fixed_model", max_seq_length=1024, dtype=None, load_in_4bit=True
)
FastLanguageModel.for_inference(model)

def chat(message, history):
    prompt = f"### Question:\n{message}\n\n### Answer:"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=300, temperature=0.7, top_p=0.9, use_cache=True)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # only return the part after ### Answer:
    ans = decoded.split("### Answer:")[-1].strip()
    ans = ans.replace("|end_of_text|>", "").replace("Source: [", "\n\nSource: [").strip()
    return ans

gr.ChatInterface(chat, title="BabyRapa v2 - Fixed Cowden Model 🍼").launch()