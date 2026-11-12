"""
Self-Supervised Pre-training on 408K dataset (Server B).
Loss = L_pseudo (D3) + L_const_self. Uses 15/16 train split.

Usage: python train_exp2_pretrain_40k.py --gpu 0
"""
import os, sys, json, argparse, time
import numpy as np, torch, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")
from config import (GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT, CONCORDE_BATCH_SIZE,
    LAMBDA_CONST, CONCORD_DATA_DIR, BENCHMARK_NAMES, SEED, PATIENCE, TRAIN_RATIO,
    BASE_INDICES, FP_INDICES, BP_INDICES, SP_INDICES)
from model import GEMModel
from dataset import load_all_benchmarks, FeatureNormalizer

GEM_DATA_DIR = "/scratch2/wl730/micro_traindata_packed_all/gem"
DEFAULT_OUTPUT = "/scratch/gem5-TAO/micro_exp/exp2/results_pretrain_528"

D3_FE = {"decode":(3873,3974),"dispatch":(3974,4075),"issue":(4075,4176),"commit":(4176,4277)}
D3_BE = {"decode":(4277,4378),"dispatch":(4378,4479),"issue":(4479,4580),"commit":(4580,4681)}
STAGE_ORDER = ["decode","dispatch","issue","commit"]
PRETRAIN_EPOCHS=1000; PRETRAIN_LR=1e-3; PRETRAIN_PATIENCE=200; LAMBDA_PSEUDO=1.0

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)
    torch.backends.cudnn.deterministic = True

def loss_const_self(mscs_pred, w_min, eps=1e-8):
    bv = mscs_pred[:, BASE_INDICES]
    cpi = 1.0/(bv * w_min.unsqueeze(1) + eps)
    return ((cpi - cpi.mean(dim=1,keepdim=True))**2).mean()

def extract_d3_pseudo_mscs(gem_raw):
    N = gem_raw.shape[0]; pseudo = np.zeros((N,16), dtype=np.float32)
    for si, st in enumerate(STAGE_ORDER):
        fs,fe = D3_FE[st]; bs,be = D3_BE[st]
        fp = np.clip(gem_raw[:,fs:fe].mean(axis=1),0,1)
        bp = np.clip(gem_raw[:,bs:be].mean(axis=1),0,1)
        tot = fp+bp; ov = tot>0.95
        if ov.any(): sc=0.95/(tot[ov]+1e-8); fp[ov]*=sc; bp[ov]*=sc
        pseudo[:,BASE_INDICES[si]]=1.0-fp-bp; pseudo[:,FP_INDICES[si]]=fp
        pseudo[:,BP_INDICES[si]]=bp; pseudo[:,SP_INDICES[si]]=0.0
    return pseudo

def create_pretrain_loader(all_data, gem_dir, batch_size=CONCORDE_BATCH_SIZE, seed=SEED):
    rng = np.random.RandomState(seed); ag,aw,ap = [],[],[]
    for name, data in sorted(all_data.items()):
        n=data["n_samples"]; idx=rng.permutation(n); tr=idx[:int(n*TRAIN_RATIO)]
        ag.append(data["gem_features"][tr]); aw.append(data["w_min"][tr])
        raw=np.load(os.path.join(gem_dir,f"{name}.npy")).astype(np.float32)
        if raw.ndim==1: raw=raw[np.newaxis,:]
        ap.append(extract_d3_pseudo_mscs(raw[tr]))
    ag=np.concatenate(ag); aw=np.concatenate(aw); ap=np.concatenate(ap)
    gn=FeatureNormalizer(); ag=gn.fit_transform(ag)
    ds=TensorDataset(torch.from_numpy(ag),torch.from_numpy(aw),torch.from_numpy(ap))
    b=min(batch_size,len(ds))
    loader=DataLoader(ds,batch_size=b,shuffle=True,num_workers=4,pin_memory=True,drop_last=False)
    print(f"  Pretrain: {len(ds):,} samples (15/16), batch={b}")
    return loader, gn

def train_pretrain(model, loader, device):
    opt=optim.AdamW(model.parameters(),lr=PRETRAIN_LR,weight_decay=1e-5)
    sch=optim.lr_scheduler.MultiStepLR(opt,[10000,14000,18000,22000],gamma=0.5)
    bl,bst,pc=float("inf"),None,0; step=0; hist=[]; t0=time.time()
    for ep in range(1,PRETRAIN_EPOCHS+1):
        model.train(); epl,elc,nb=0,0,0
        for gf,wm,pt in loader:
            gf,wm,pt=gf.to(device),wm.to(device),pt.to(device)
            pred=model(gf); lp=F.mse_loss(pred,pt); lc=loss_const_self(pred,wm)
            loss=LAMBDA_PSEUDO*lp+LAMBDA_CONST*lc
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sch.step(); step+=1; epl+=lp.item(); elc+=lc.item(); nb+=1
        al=(LAMBDA_PSEUDO*epl+LAMBDA_CONST*elc)/nb
        hist.append({"epoch":ep,"step":step,"loss":al,"L_pseudo":epl/nb,"L_const_self":elc/nb})
        if al<bl: bl=al; bst={k:v.cpu().clone() for k,v in model.state_dict().items()}; pc=0
        else:
            pc+=1
            if pc>=PRETRAIN_PATIENCE: print(f"  Early stop ep {ep}"); break
        if ep%10==0: print(f"  ep {ep} | loss={al:.6f} | {time.time()-t0:.0f}s")
    model.load_state_dict(bst)
    print(f"  Done. Best={bl:.6f} ({time.time()-t0:.0f}s)")
    return model, hist

def main():
    pa=argparse.ArgumentParser(); pa.add_argument("--gpu",type=int,default=0)
    pa.add_argument("--output_dir",type=str,default=DEFAULT_OUTPUT)
    args=pa.parse_args(); set_seed(SEED)
    device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}\nData: {GEM_DATA_DIR}\nOutput: {args.output_dir}")
    all_data=load_all_benchmarks(GEM_DATA_DIR,CONCORD_DATA_DIR,BENCHMARK_NAMES)
    loader,gn=create_pretrain_loader(all_data,GEM_DATA_DIR)
    model=GEMModel(GEM_INPUT_DIM,GEM_HIDDEN_DIMS,DROPOUT).to(device)
    model,hist=train_pretrain(model,loader,device)
    torch.save(model.state_dict(),os.path.join(args.output_dir,"pretrained_40k.pt"))
    np.savez(os.path.join(args.output_dir,"gem_normalizer_40k.npz"),mean=gn.mean,std=gn.std)
    with open(os.path.join(args.output_dir,"pretrain_history_40k.json"),"w") as f: json.dump(hist,f,indent=2)
    print("Done!")

if __name__=="__main__": main()
