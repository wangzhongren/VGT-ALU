import base64

with open("vgt_model.onnx", "rb") as f:
    encoded_string = base64.b64encode(f.read()).decode('utf-8')
    
with open("model_base64.txt", "w") as f:
    f.write(encoded_string)

print("✅ Base64 字符串已保存至 model_base64.txt")