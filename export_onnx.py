import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 必须保留与训练时完全一致的模型定义
class VGTProModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(10, hidden_size)
        self.reducer = nn.Conv1d(2 * hidden_size, hidden_size, kernel_size=1)
        self.conv_process = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.output_proj = nn.Conv1d(hidden_size, 10, kernel_size=1)

    def forward(self, x):
        B, L = x.shape
        digits = L // 2
        x_emb = self.embedding(x).transpose(1, 2)
        a_part = x_emb[:, :, :digits]; b_part = x_emb[:, :, digits:]
        
        h = torch.relu(self.reducer(torch.cat([a_part, b_part], dim=1)))
        h = nn.functional.pad(h, (0, 1)) 
        
        # 这里的循环在导出 ONNX 时会被展开(Unroll)
        # 建议导出时针对最大位数（比如 20 位）设定足够的循环次数
        # 20位加法 L=20, digits=20, h.size(2) 就是 21
        loop_count = h.size(2) + 2 
        
        for i in range(loop_count):
            dilation = 1 if i < 4 else (2 if i < 8 else 4)
            padding = dilation
            # 注意：ONNX 导出 F.conv1d 比较稳妥的方式是使用实例化的层，
            # 但你训练时用了 self.conv_process 的参数，这里保持一致：
            h_residual = F.conv1d(h, self.conv_process.weight, self.conv_process.bias, 
                                  padding=padding, dilation=dilation)
            h = torch.relu(h_residual) + h 
            
        return self.output_proj(h).transpose(1, 2)

# 2. 加载权重并导出
def convert():
    hidden_size = 128  # 保持与训练一致
    model = VGTProModel(hidden_size)
    
    # 加载权重
    checkpoint = torch.load("vgt_pro_logic_machine.pth", map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # 构造模拟输入：假设我们要支持到 20 位加法，输入长度为 40
    # 使用 dynamic_axes 可以让模型处理不同长度的输入
    dummy_input = torch.randint(0, 10, (1, 40)) 

    torch.onnx.export(
        model,
        dummy_input,
        "vgt_model.onnx",
        export_params=True,
        opset_version=12,  #  dilation 逻辑需要较新的 opset
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {1: 'sequence_length'},
            'output': {1: 'out_sequence_length'}
        }
    )
    print("✅ 成功导出 vgt_model.onnx")

if __name__ == "__main__":
    convert()