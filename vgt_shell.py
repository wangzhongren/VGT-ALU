import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# ==========================================
# 1. Physical Layer: VGT-Pro Core Architecture
# ==========================================
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

# ==========================================
# 2. Logic Layer: Neural Arithmetic Logic Unit
# ==========================================
class NeuralALU:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = VGTProModel().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def add(self, a, b):
        max_len = max(len(str(a)), len(str(b))) + 1
        a_d = [int(d) for d in str(a).zfill(max_len)][::-1]
        b_d = [int(d) for d in str(b).zfill(max_len)][::-1]
        x_in = torch.tensor([a_d + b_d], dtype=torch.long).to(self.device)
        with torch.no_grad():
            preds = self.model(x_in)[0].argmax(dim=-1).cpu().tolist()
        return sum(d * (10 ** i) for i, d in enumerate(preds))

    def sub(self, a, b):
        if a < b: return -self.sub(b, a)
        digits = len(str(a))
        b_10_comp = int("".join([str(9 - int(d)) for d in str(b).zfill(digits)])) + 1
        raw_res = self.add(a, b_10_comp)
        return int(str(raw_res)[-digits:])

    def mul(self, a, b):
        res = 0
        for i, digit in enumerate(reversed(str(b))):
            partial = 0
            for _ in range(int(digit)): partial = self.add(partial, a)
            res = self.add(res, partial * (10 ** i))
        return res

# ==========================================
# 3. Application Layer: Interactive Shell
# ==========================================
def start_shell(model_path):
    alu = NeuralALU(model_path)
    print("\n" + "="*50)
    print("� VGT-Pro Neural Arithmetic Shell")
    print("Commands: (e.g. 123+456, 1000-45, 12*12) | 'exit' to quit")
    print("="*50)

    while True:
        expr = input(">>> ").replace(" ", "")
        if expr.lower() == 'exit': break
        try:
            tokens = re.split(r'([\+\-\*])', expr)
            a, op, b = int(tokens[0]), tokens[1], int(tokens[2])
            if op == '+': res = alu.add(a, b)
            elif op == '-': res = alu.sub(a, b)
            elif op == '*': res = alu.mul(a, b)
            print(f"Result: {res}")
        except Exception as e:
            print(f"Error: Format 'A op B' required.")

if __name__ == "__main__":
    start_shell("vgt_pro_logic_machine.pth")