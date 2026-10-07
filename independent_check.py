"""Independent arithmetic check of published weights, no retraining."""
import json,time,random,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from vgt_alu_core import VGTProModel,NeuralALU

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
checkpoint=torch.load(ROOT/'vgt_pro_logic_machine.pth',map_location='cpu',weights_only=True)
model=VGTProModel(checkpoint['hidden_size']).to(DEVICE).eval()
model.load_state_dict(checkpoint['model_state_dict'],strict=True)
rng=random.Random(20261007)

@torch.inference_mode()
def predict(pairs,digits,extra_steps):
    x=torch.tensor([[int(v) for v in str(a).zfill(digits)[::-1]+str(b).zfill(digits)[::-1]] for a,b in pairs],device=DEVICE)
    emb=model.embedding(x).transpose(1,2)
    h=torch.relu(model.reducer(torch.cat([emb[:,:,:digits],emb[:,:,digits:]],1)))
    h=F.pad(h,(0,1))
    for i in range(h.shape[2]+extra_steps):
        dilation=1 if i<4 else (2 if i<8 else 4)
        h=h+torch.relu(F.conv1d(h,model.conv_process.weight,model.conv_process.bias,padding=dilation,dilation=dilation))
    logits=model.output_proj(h).transpose(1,2)
    pred=logits.argmax(-1).cpu().tolist()
    return [sum(v*10**i for i,v in enumerate(row)) for row in pred],bool(torch.isfinite(logits).all()),float(h.abs().max())

def assess(pairs,digits,extra_steps):
    errors=[];correct=0;finite=True;peak=0.
    for offset in range(0,len(pairs),64):
        batch=pairs[offset:offset+64];pred,ok,hmax=predict(batch,digits,extra_steps);finite&=ok;peak=max(peak,hmax)
        for (a,b),value in zip(batch,pred):
            correct+=int(value==a+b)
            if value!=a+b and len(errors)<3:errors.append(dict(a=str(a),b=str(b),expected=str(a+b),predicted=str(value)))
    return dict(n=len(pairs),correct=correct,accuracy=correct/len(pairs),all_logits_finite=finite,max_abs_hidden=peak,first_errors=errors)

def main():
    started=time.time();results=[]
    for digits in [1,3,6,12,20,30,64,128]:
        n=1000 if digits<=30 else 200
        pairs=[(rng.randrange(10**(digits-1),10**digits),rng.randrange(10**(digits-1),10**digits)) for _ in range(n)]
        for mode,width,steps in [('training_forward',digits,2),('core_direct_forward',digits,4),('public_add_wrapper',digits+1,4)]:
            result=dict(digits=digits,mode=mode,**assess(pairs,width,steps));results.append(result)
            print('RANDOM',digits,mode,result['accuracy'],result['all_logits_finite'],flush=True)
    carry=[]
    for digits in [1,6,20,30,64,128]:
        pairs=[(10**digits-1,1),(10**digits-1,10**digits-1),(int('9'*(digits-1)+'8'),2),(0,0),(10**(digits-1),0)]
        carry.append(dict(digits=digits,mode='public_add_wrapper',**assess(pairs,digits+1,4)))
    # Exact single-digit domain exhaustively, not just random samples.
    exhaustive=assess([(a,b) for a in range(10) for b in range(10)],2,4)
    alu=NeuralALU.__new__(NeuralALU);alu.device=torch.device(DEVICE);alu.model=model
    operations={}
    for op in ['sub','mul','compare']:
        rows=[]
        pairs=[(50000,12345),(0,0),(0,123),(123,0),(100,200),(1234,5678),(10**20-1,1),(12345678901234567890,98765432109876543210)]
        pairs+= [(rng.randrange(10**6),rng.randrange(10**6)) for _ in range(30)]
        for a,b in pairs:
            expected=a-b if op=='sub' else (a*b if op=='mul' else a>=b)
            actual=getattr(alu,op)(a,b)
            rows.append(dict(a=str(a),b=str(b),expected=str(expected),actual=str(actual),correct=actual==expected))
        operations[op]=dict(n=len(rows),correct=sum(r['correct'] for r in rows),rows=rows)
        print('OP',op,operations[op]['correct'],'/',len(rows),flush=True)
    negative=[]
    for op,a,b in [('add',-1,2),('mul',-2,3),('mul',2,-3)]:
        try:negative.append(dict(op=op,a=a,b=b,value=getattr(alu,op)(a,b)))
        except Exception as exc:negative.append(dict(op=op,a=a,b=b,error=type(exc).__name__+': '+str(exc)))
    result=dict(seed=20261007,device=DEVICE,torch=str(torch.__version__),parameters=sum(p.numel() for p in model.parameters()),weight_bytes=(ROOT/'vgt_pro_logic_machine.pth').stat().st_size,weight_sha256=hashlib.sha256((ROOT/'vgt_pro_logic_machine.pth').read_bytes()).hexdigest(),checkpoint_metadata={k:v for k,v in checkpoint.items() if k!='model_state_dict'},random_addition=results,carry_chain_cases=carry,single_digit_exhaustive=exhaustive,orchestrated_operations=operations,negative_inputs=negative,elapsed_seconds=time.time()-started,notes=['Exact whole-integer equality, not per-digit accuracy.','No training and no checkpoint tuning.','Public add pads one extra leading zero; training forward uses two fewer iterations than core.','Subtraction/multiplication/comparison contain explicit host-language algorithms.','Finite samples do not prove all-input correctness. Negative operands outside documented nonnegative input mechanism are tested separately.'])
    (ROOT/'independent-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',result['parameters'],result['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
