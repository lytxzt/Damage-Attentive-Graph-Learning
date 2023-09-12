import openai

# 设置你的 API 密钥
api_key = "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"

# 输入代码示例
code_input = "explain the code:\n"
with open("./Main_algorithm_GCN/CR_MGC.py", 'r') as f:
    code = f.read()

code_input += code

# 调用 ChatGPT API
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=code_input,
    max_tokens=200,  # 设置生成的最大令牌数
    api_key=api_key
)

# 解析生成的代码
parsed_code = response.choices[0].text
print(parsed_code)

