"""Frozen original-weight probes: local rules, controlled chains, input influence."""
import hashlib
import json
from pathlib import Path
import random
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vgt_alu_core import VGTProModel

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WEIGHTS = ROOT/'vgt_pro_logic_machine.pth'
assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == '790b1409481782e43dc97822b558312a74377b0c9ad285efe9148e997ae84271'
checkpoint = torch.load(WEIGHTS,map_location='cpu',weights_only=True)
model = VGTProModel(checkpoint['hidden_size']).to(DEVICE).eval()
model.load_state_dict(checkpoint['model_state_dict'],strict=True)


@torch.inference_mode()
def predict(pairs,width):
    result = []
    for offset in range(0,len(pairs),64):
        x = torch.tensor([[int(c) for c in str(a).zfill(width)[::-1]+str(b).zfill(width)[::-1]] for a,b in pairs[offset:offset+64]],device=DEVICE)
        result.extend(model(x).argmax(-1).cpu().tolist())
    return result


def number(digits): return sum(v*10**i for i,v in enumerate(digits))


def main():
    out = dict(weight_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),device=DEVICE,seed=20261013,notes=['All probes use frozen original weights and original public inference schedule.','No training or weight edits. Hidden-state change measures causal influence, not a decoded carry bit.','Controlled chain comparisons keep width and iteration count fixed.'])
    # All possible inputs to a decimal full-adder in a two-digit context.
    pairs = []
    labels = []
    for a in range(10):
        for b in range(10):
            for incoming in [0,1]:
                pairs.append((a*10+(9 if incoming else 0),b*10+(1 if incoming else 0)))
                labels.append((a,b,incoming))
    preds = predict(pairs,3)
    local = {k:dict(n=0,digit_correct=0,carry_out_correct=0) for k in ['kill','propagate','generate']}
    for (a,b,c),pred in zip(labels,preds):
        k = 'kill' if a+b<=8 else ('propagate' if a+b==9 else 'generate')
        local[k]['n'] += 1
        local[k]['digit_correct'] += pred[1] == (a+b+c)%10
        local[k]['carry_out_correct'] += pred[2] == (a+b+c)//10
    out['local_full_adder_200_cases'] = local
    print('LOCAL',local,flush=True)

    # Change only one low input digit, all higher digit sums in the chain are nine.
    rng = random.Random(20261013)
    width = 65
    chain_rows = []
    for length in [0,1,2,3,4,5,6,8,12,16,24,32,48]:
        pairs = []
        layouts = []
        for _ in range(128):
            ad = [rng.randrange(10) for _ in range(64)]
            bd = [rng.randrange(10) for _ in range(64)]
            start = rng.randrange(64-length-1)
            endpoint = start+length+1
            ad[start] = rng.randrange(1,9)
            low_b = rng.randrange(9-ad[start])
            high_b = rng.randrange(10-ad[start],10)
            for p in range(start+1,endpoint): bd[p] = 9-ad[p]
            ad[endpoint] = bd[endpoint] = 0
            a = number(ad)
            bd[start] = low_b
            pairs.append((a,number(bd)))
            bd[start] = high_b
            pairs.append((a,number(bd)))
            layouts.append((start,endpoint))
        pred = predict(pairs,width)
        row = dict(chain_length=length,width=width,iterations=width+1+4,n_pairs=128,
                   endpoint_carry0_correct=0,endpoint_carry1_correct=0,
                   both_endpoints_correct=0,entire_chain_correct=0,whole_integer_correct=0)
        for j,(start,end) in enumerate(layouts):
            p0,p1 = pred[2*j],pred[2*j+1]
            c0,c1 = p0[end]==0,p1[end]==1
            row['endpoint_carry0_correct'] += c0
            row['endpoint_carry1_correct'] += c1
            row['both_endpoints_correct'] += c0 and c1
            row['entire_chain_correct'] += all(p0[k]==9 and p1[k]==0 for k in range(start+1,end)) and c0 and c1
            for index in [2*j,2*j+1]:
                a,b = pairs[index]
                row['whole_integer_correct'] += number(pred[index])==a+b
        chain_rows.append(row)
        print('CHAIN',row,flush=True)
    out['controlled_chains'] = chain_rows

    # Does changing the units input still influence hidden states at distant positions?
    a = 10**20-1
    width = 21
    x = torch.tensor([[int(c) for c in str(a).zfill(width)[::-1]+str(b).zfill(width)[::-1]] for b in [0,1]],device=DEVICE)
    with torch.inference_mode():
        emb = model.embedding(x).transpose(1,2)
        h = F.pad(torch.relu(model.reducer(torch.cat([emb[:,:,:width],emb[:,:,width:]],1))),(0,1))
        snapshots = []
        for iteration in range(209):
            if iteration in [0,1,4,8,9,10,16,26,52,104,208]:
                diff = (h[1]-h[0]).norm(dim=0).cpu().tolist()
                digits = model.output_proj(h).argmax(1).cpu().tolist()
                snapshots.append(dict(iteration=iteration,hidden_changed_positions=[i for i,v in enumerate(diff) if v>1e-6],hidden_difference_norms=diff,zero_result=str(number(digits[0])),one_result=str(number(digits[1])),zero_correct=number(digits[0])==a,one_correct=number(digits[1])==a+1))
            if iteration==208: break
            dilation = 1 if iteration<4 else (2 if iteration<8 else 4)
            h = h+torch.relu(F.conv1d(h,model.conv_process.weight,model.conv_process.bias,padding=dilation,dilation=dilation))
    out['units_input_intervention'] = snapshots
    for row in snapshots:
        print('INFLUENCE',row['iteration'],row['hidden_changed_positions'],'correct',row['zero_correct'],row['one_correct'],flush=True)
    (ROOT/'understanding'/'original-learning-probe.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__': main()
