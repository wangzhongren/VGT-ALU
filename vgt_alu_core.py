import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. VGT-Pro 核心架构 ---
class VGTProModel(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(10, hidden_size)
        self.reducer = nn.Conv1d(2 * hidden_size, hidden_size, kernel_size=1)
        self.conv_process = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.output_proj = nn.Conv1d(hidden_size, 10, kernel_size=1)

    def forward(self, x):
        digits = x.shape[1] // 2
        x_emb = self.embedding(x).transpose(1, 2)
        h = torch.relu(self.reducer(torch.cat([x_emb[:, :, :digits], x_emb[:, :, digits:]], dim=1)))
        h = F.pad(h, (0, 1)) 
        for i in range(h.size(2) + 4): 
            dilation = 1 if i < 4 else (2 if i < 8 else 4)
            h_residual = F.conv1d(h, self.conv_process.weight, self.conv_process.bias, 
                                  padding=dilation, dilation=dilation)
            h = torch.relu(h_residual) + h 
        return self.output_proj(h).transpose(1, 2)

# --- 2. NeuralALU 底层单元封装 ---
class NeuralALU:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = VGTProModel().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"[*] VGT-Pro NALU Core Initialized on {self.device}")

    def add(self, a, b):
        """原生逻辑：加法"""
        max_len = max(len(str(a)), len(str(b))) + 1
        a_d = [int(d) for d in str(a).zfill(max_len)][::-1]
        b_d = [int(d) for d in str(b).zfill(max_len)][::-1]
        x_in = torch.tensor([a_d + b_d], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x_in)
            preds = logits[0].argmax(dim=-1).cpu().tolist()
        
        return sum(d * (10 ** i) for i, d in enumerate(preds))

    def sub(self, a, b):
        """迁移逻辑：基于补码的减法 (A - B)"""
        if a < b:
            return -self.sub(b, a) # 处理负数结果
        
        digits = len(str(a))
        # 计算 10 的补码
        b_str = str(b).zfill(digits)
        b_9_comp = "".join([str(9 - int(d)) for d in b_str])
        b_10_comp = int(b_9_comp) + 1
        
        # 利用加法核心计算补码和
        raw_res = self.add(a, b_10_comp)
        # 补码减法法则：丢弃最高位进位
        return int(str(raw_res)[-digits:])

    def mul(self, a, b):
        """编排逻辑：基于递归累加的乘法"""
        res = 0
        b_str = str(b)
        for i, digit in enumerate(reversed(b_str)):
            partial_sum = 0
            for _ in range(int(digit)):
                partial_sum = self.add(partial_sum, a)
            res = self.add(res, partial_sum * (10 ** i))
        return res

    def compare(self, a, b):
        """逻辑判定：A >= B ?"""
        # 利用减法特性进行判定
        diff = self.sub(a, b)
        return diff >= 0

# --- 3. 运行验证 ---
if __name__ == "__main__":
    # 使用你的模型权重文件
    alu = NeuralALU("vgt_pro_logic_machine.pth")
    
    print("\n--- NALU 指令集测试 ---")
    
    # 测试加法 (30位外推)
    v1, v2 = 12345678901234567890, 98765432109876543210
    print(f"[ADD] {v1} + {v2} = {alu.add(v1, v2)}")
    
    # 测试减法 (补码逻辑)
    s1, s2 = 50000, 12345
    print(f"[SUB] {s1} - {s2} = {alu.sub(s1, s2)}")
    
    # 测试乘法 (算法递归)
    m1, m2 = 1234, 5678
    print(f"[MUL] {m1} * {m2} = {alu.mul(m1, m2)}")
    
    # 测试逻辑比较
    c1, c2 = 100, 200
    print(f"[COMP] {c1} >= {c2} is {alu.compare(c1, c2)}")