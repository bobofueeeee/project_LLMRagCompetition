from transformers import AutoModelForCausalLM, AutoTokenizer

# 假设你已经有了 LLaMA-3 的模型权重和配置文件，并放置在正确的路径下
model_name_or_path = "D:/wk/llama3-Chinese-chat-8b"  # 替换为你的 LLaMA-3 模型路径
tokenizer_name_or_path = "D:/wk/llama3-Chinese-chat-8b"  # 如果有单独的 tokenizer 路径，否则可以与 model 相同

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# 输入文本
input_text = "Hello, my name is "


# 对输入文本进行编码
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 生成文本（这里只是一个简单的示例，你可能需要调整参数以满足你的需求）
generated_text = model.generate(
    input_ids,
    max_length=50,  # 生成的最大长度
    num_beams=5,  # 使用 beam search 的数量
    temperature=1.0,  # 控制随机性的参数
    top_k=50,  # 控制生成的词汇范围的参数
    top_p=0.95,  # 控制生成的词汇多样性的参数
    pad_token_id=tokenizer.eos_token_id,  # 填充 token 的 ID，通常是 EOS token
    eos_token_id=tokenizer.eos_token_id,  # 结束 token 的 ID，也是 EOS token
)

# 将生成的 IDs 解码为文本
output_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)

print(output_text)