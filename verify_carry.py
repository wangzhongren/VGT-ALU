"""Recheck counterexamples with the original public API and extra iterations."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from vgt_alu_core import VGTProModel, NeuralALU

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
state = torch.load(ROOT / 'vgt_pro_logic_machine.pth', map_location='cpu', weights_only=True)

@torch.inference_mode()
def longer_forward(model, device, a, b, multiplier):
    width = max(len(str(a)), len(str(b))) + 1
    x = torch.tensor([[int(v) for v in str(a).zfill(width)[::-1] + str(b).zfill(width)[::-1]]], device=device)
    emb = model.embedding(x).transpose(1, 2)
    h = F.pad(torch.relu(model.reducer(torch.cat([emb[:, :, :width], emb[:, :, width:]], 1))), (0, 1))
    steps = multiplier * (h.shape[2] + 4)
    for i in range(steps):
        dilation = 1 if i < 4 else (2 if i < 8 else 4)
        h = h + torch.relu(F.conv1d(h, model.conv_process.weight, model.conv_process.bias, padding=dilation, dilation=dilation))
    logits = model.output_proj(h).transpose(1, 2)
    pred = logits[0].argmax(-1).cpu().tolist()
    actual = sum(v * 10**i for i, v in enumerate(pred))
    return dict(iterations=steps, expected=str(a+b), actual=str(actual), correct=actual == a+b, finite=bool(torch.isfinite(logits).all()))

def main():
    result = {}
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        model = VGTProModel(state['hidden_size']).to(device).eval()
        model.load_state_dict(state['model_state_dict'], strict=True)
        alu = NeuralALU.__new__(NeuralALU)
        alu.device = torch.device(device)
        alu.model = model
        pairs = [(10**20-1, 1), (858840639901995615436343696906, 396459140098994922466153244359)]
        rows = []
        for a, b in pairs:
            actual = alu.add(a, b)
            rows.append(dict(a=str(a), b=str(b), expected=str(a+b), actual=str(actual), correct=actual == a+b))
        sweep = []
        for digits in range(1, 81):
            a = 10**digits-1
            actual = alu.add(a, 1)
            sweep.append(dict(digits=digits, correct=actual == a+1, actual=str(actual)))
        extended = {str(digits): [longer_forward(model, device, 10**digits-1, 1, k) for k in [1, 2, 4, 8]] for digits in [20, 30, 64]}
        result[device] = dict(public_api_recheck=rows, nines_plus_one=sweep, extended_iterations=extended)
        print(device, 'recheck', rows, flush=True)
        print(device, 'failed_nines_lengths', [x['digits'] for x in sweep if not x['correct']], flush=True)
        print(device, 'extended', extended, flush=True)
    (ROOT / 'carry-verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
