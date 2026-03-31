# coding=utf-8
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from time import time



def derta_time(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        user_input = input('you: ')
        result = func(user_input = user_input,*args, **kwargs)
        end_time = time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

model_name = "./models/Qwen3.5-0.8B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map= "auto" ,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 系统提示（永久保留）
system_prompt = {
    "role": "system",
    "content": """
    你的名字是Qekys.主要使用中文回答,简洁明了.
    看不懂用户输入的内容时,使用俏皮的话回复.
    【人称规则】
    - 用户说"我"=用户自己，你说"我"=你自己(Qekys)
    - 称呼用户用"你"，称呼自己用"我"
    """
}
messages = [system_prompt]

# 【记忆限制】最多保留4次交互
MAX_INTERACTIONS = 20

@derta_time
def chat(user_input: str = None):
    global messages
    response = ""
    # user = input('you:')

    # 添加用户消息
    messages.append({"role": "user", "content": user_input})
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    print("Qekys: ", end="", flush=True)
    
    for new_text in streamer:
        print(new_text, end="", flush=True)
        response += new_text
    
    messages.append({"role": "assistant", "content": response})
    
    # 【记忆限制】保留 system + 最后N次交互
    if len(messages) > 1 + MAX_INTERACTIONS * 2:
        messages = [messages[0]] + messages[-MAX_INTERACTIONS * 2:]
    
    print()
    
if __name__ == '__main__':
    while True:
        print("=" * 50)
        chat()