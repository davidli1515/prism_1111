"""
Pretrain+FT on 408K dataset (Server B). FT only, loads pretrained_40k.pt.
Results: exp5_{ratio}pct_40k.json in same output dir.

Usage: python train_exp5_pretrain_ft_40k.py --gpu 0 --ratio 0.05
"""
import os, sys, json, argparse, time, hashlib
import numpy as np, torch, torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")
from config import (GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT, BATCH_SIZE,
    CONCORDE_BATCH_SIZE, LAMBDA_CONST, LAMBDA_MONO,
    CONCORD_DATA_DIR, BENCHMARK_NAMES, SEED, TRAIN_RATIO)
from model import GEMModel, compute_cpi_mean, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, FeatureNormalizer, GEMDataset

GEM_DATA_DIR = "/data/micro_traindata_packed_all/gem"
PRETRAIN_DIR = "/path/to/gem5/micro_exp/exp2/results_pretrain_run2"
FT_LR=5e-4; FT_EPOCHS=1500; FT_PATIENCE=200

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)
    torch.backends.cudnn.deterministic = True

def stable_hash(name):
    return int(hashlib.md5(name.encode()).hexdigest(),16)%(2**31)

def compute_mape(p,t,eps=1e-8):
    return (torch.abs(p-t)/(torch.abs(t)+eps)*100).mean().item()

@torch.no_grad()
def evaluate_gem(model, loader, device):
    model.eval(); P,G=[],[]
    for b in loader:
        m=model(b["gem_features"].to(device))
        P.append(compute_cpi_mean(m,b["w_min"].to(device)).cpu()); G.append(b["cpi_golden"])
    return compute_mape(torch.cat(P),torch.cat(G))

def create_labeled_split(all_data, labeled_ratio, gem_norm, batch_size=CONCORDE_BATCH_SIZE, seed=SEED):
    rng=np.random.RandomState(seed)
    tg,tc,tm,tp,tw=[],[],[],[],[]
    eg,ec,em,ep_,ew=[],[],[],[],[]
    for name,data in sorted(all_data.items()):
        n=data["n_samples"]; idx=rng.permutation(n)
        split=int(n*TRAIN_RATIO); tr_idx,te_idx=idx[:split],idx[split:]
        n_labeled=max(1,int(len(tr_idx)*labeled_ratio))
        lrng=np.random.RandomState(seed+stable_hash(name)+int(round(labeled_ratio*1000)))
        labeled_idx=tr_idx[lrng.permutation(len(tr_idx))[:n_labeled]]
        tg.append(data["gem_features"][labeled_idx]); tc.append(data["concord_features"][labeled_idx])
        tm.append(data["mscs_target"][labeled_idx]); tp.append(data["cpi_golden"][labeled_idx])
        tw.append(data["w_min"][labeled_idx])
        eg.append(data["gem_features"][te_idx]); ec.append(data["concord_features"][te_idx])
        em.append(data["mscs_target"][te_idx]); ep_.append(data["cpi_golden"][te_idx])
        ew.append(data["w_min"][te_idx])
    tg=gem_norm.transform(np.concatenate(tg)); eg=gem_norm.transform(np.concatenate(eg))
    tc=np.concatenate(tc); ec=np.concatenate(ec)
    tm=np.concatenate(tm); em=np.concatenate(em)
    tp=np.concatenate(tp); ep_=np.concatenate(ep_)
    tw=np.concatenate(tw); ew=np.concatenate(ew)
    tds=GEMDataset(tg,tc,tm,tp,tw); eds=GEMDataset(eg,ec,em,ep_,ew)
    bs=min(batch_size,len(tds))
    tl=DataLoader(tds,batch_size=bs,shuffle=True,num_workers=4,pin_memory=True,
                  drop_last=len(tds)>bs)
    tel=DataLoader(eds,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    print(f"  Ratio: {labeled_ratio:.0%} | Train: {len(tds):,} | Test: {len(eds):,}")
    return tl, tel

def finetune(model, tl, tel, device, label="PT-FT"):
    opt=optim.AdamW(model.parameters(),lr=FT_LR,weight_decay=1e-5)
    sch=optim.lr_scheduler.MultiStepLR(opt,[5000,8000,12000,16000],gamma=0.5)
    bm,bs,pc=float("inf"),None,0; step=0; hist=[]
    for ep in range(1,FT_EPOCHS+1):
        model.train(); el,nb=0,0
        for b in tl:
            feat=b["gem_features"].to(device); pred=model(feat)
            loss=loss_task(pred,b["mscs_target"].to(device))+\
                 LAMBDA_CONST*loss_const(pred,b["w_min"].to(device),b["cpi_golden"].to(device))+\
                 LAMBDA_MONO*loss_mono(pred)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sch.step(); step+=1; el+=loss.item(); nb+=1
        mape=evaluate_gem(model,tel,device)
        hist.append({"epoch":ep,"step":step,"mape":mape,"loss":el/nb})
        if mape<bm: bm=mape; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}; pc=0
        else:
            pc+=1
            if pc>=FT_PATIENCE: print(f"  [{label}] Early stop ep {ep} (best={bm:.2f}%)"); break
        if ep%50==0: print(f"  [{label}] ep {ep} | MAPE={mape:.2f}%")
    model.load_state_dict(bs)
    return model, bm, hist

def main():
    pa=argparse.ArgumentParser(); pa.add_argument("--gpu",type=int,default=0)
    pa.add_argument("--ratio",type=float,required=True)
    pa.add_argument("--pretrain_dir",type=str,default=PRETRAIN_DIR)
    args=pa.parse_args()
    set_seed(SEED)
    device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ckpt=os.path.join(args.pretrain_dir,"pretrained_40k.pt")
    norm_path=os.path.join(args.pretrain_dir,"gem_normalizer_40k.npz")
    assert os.path.exists(ckpt), f"Not found: {ckpt}"
    assert os.path.exists(norm_path), f"Not found: {norm_path}"
    ratio_tag=f"{int(args.ratio*100)}pct"
    print(f"Device: {device} | Ratio: {args.ratio:.0%}")
    gem_norm=FeatureNormalizer(); nd=np.load(norm_path)
    gem_norm.mean,gem_norm.std=nd["mean"],nd["std"]
    all_data=load_all_benchmarks(GEM_DATA_DIR,CONCORD_DATA_DIR,BENCHMARK_NAMES)
    tl,tel=create_labeled_split(all_data,args.ratio,gem_norm)
    model=GEMModel(GEM_INPUT_DIM,GEM_HIDDEN_DIMS,DROPOUT).to(device)
    model.load_state_dict(torch.load(ckpt,map_location=device))
    zero=evaluate_gem(model,tel,device); print(f"  Zero-shot: {zero:.2f}%")
    t0=time.time()
    model,mape,hist=finetune(model,tl,tel,device,label=f"PT-FT-{ratio_tag}")
    result={"labeled_ratio":args.ratio,"zero_shot":zero,"pretrain_ft":mape,"history":hist}
    out_path=os.path.join(args.pretrain_dir,f"exp5_{ratio_tag}.json")
    with open(out_path,"w") as f: json.dump(result,f,indent=2)
    torch.save(model.state_dict(),os.path.join(args.pretrain_dir,f"pretrain_ft_{ratio_tag}.pt"))
    print(f"\n  {zero:.2f}% -> {mape:.2f}% ({time.time()-t0:.0f}s)\n  Saved: {out_path}")

if __name__=="__main__": main()
