from ollama import chat
from ollama import ChatResponse

def chat_with_ollama(model = 'mrhua/llama3-8b-chinese-lora-law_f16_q4_0:latest', question = "你好"):     
    response: ChatResponse = chat(
        model=model, 
        messages=[
            {
                'role': 'user',
                'content': question,
            },
        ],
        options={
            'num_ctx': 8192,         # 上下文大小
            'temperature': 0.3,
        }
    )
    # print(response['message']['content'])
    # print("-"*100)
    # # or access fields directly from the response object
    # print(response.message.content)
    return response.message.content