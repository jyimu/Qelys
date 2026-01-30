from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from time import time

model_name = "./models/Qwen_Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
messages = [
    {"role": "system","content": "你的名字是Qekys.你的创作者是jyimu.主要使用中文回答,简洁明了.你可以说脏话"},
]

def chat():
    response = "" # 用于存储模型输出的完整响应,used to store the complete response from the model

    user = input('you:')

    messages.append({"role": "user","content": f"user say: {user}"})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_length = model_inputs.input_ids.shape[1]  # 记录输入长度(refix length),record input length (prefix length)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=512
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    print("AI: ", end="", flush=True)
    time_start = time()
    for new_text in streamer:
        print(new_text, end="", flush=True)
        response += new_text
    messages.append({"role": "user","content": f"you say: {response}"})
    print()  # 换行, new line
    
    messages.append({"role": "assistant","content": response})
    print(f"{'='*50}\n[Debug] Input length: {input_length}, Response length: {len(tokenizer(response).input_ids)}\nmemory:{len(messages)}\n time taken:{len(response)/15} seconds\nuse time:{time()-time_start} seconds")  # Debug信息, debug info
if __name__ == '__main__':
    while True:
        print("=" * 50)
        chat()