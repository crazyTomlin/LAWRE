from transformers import AutoModelForCausalLM, AutoTokenizer

# 全局变量，用于缓存模型和分词器
_model = None
_tokenizer = None
_device = "cuda"

def initialize_model(model_path="/root/yangyi/pretrained_model/LawLLM-7B"):
    """初始化模型和分词器"""
    global _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        print("正在加载模型和分词器...")
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 如果 tokenizer 没有 pad_token，就将 eos_token 设置为 pad_token
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        
        print("模型和分词器加载完成")

def chat_with_hf(model_name=None, question="你好", max_new_tokens=1024, temperature=0.3):
    """
    使用HuggingFace模型进行对话
    
    Args:
        model_name: 模型名称（暂时未使用，保持接口一致性）
        question: 用户问题
        max_new_tokens: 最大生成token数
        temperature: 生成温度
    
    Returns:
        包含生成内容的响应对象
    """
    
    # 构建消息
    messages = [
        {"role": "system", "content": "你是LawLLM，一个由复旦大学DISC实验室创造的法律助手。"},
        {"role": "user", "content": question}
    ]
    
    # 应用聊天模板
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    model_inputs = _tokenizer([text], return_tensors="pt", padding=True).to(_device)
    
    # 生成响应
    generated_ids = _model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
        pad_token_id=_tokenizer.eos_token_id
    )
    
    # 解码生成的内容
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response_content = _tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 创建类似ollama响应格式的对象
    class MockMessage:
        def __init__(self, content):
            self.content = content
    
    class MockResponse:
        def __init__(self, content):
            self.message = MockMessage(content)
    
    return MockResponse(response_content)


if __name__ == "__main__":
    initialize_model()
    response = chat_with_hf(model_name='LawLLM-7B', question="你好")
    print(response.message.content)
    