import os
import gc
import sys
import glob
import shutil
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, log_loss
from tqdm.auto import tqdm



class Config:
    SEED = 42
    MAX_LEN = 128
    BATCH_SIZE = 16
    VAL_SPLIT = 0.10
    
    LABELS_S2 = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
    LABELS_S3 = ['stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy', 'invalidation']
    
    OUTPUT_ROOT = "./10_technique_submissions"

# Path Finder
def find_model_path(filename, search_hint=None):
    exact_candidates = [
        f"../input/{search_hint}/{filename}" if search_hint else "",
        f"../input/task9-benchmarks/{filename}",
        f"../input/load-polar-4bestmodels/final_best_models/{filename}",
        f"../input/task9_semeval3_new/{filename}"
    ]
    for p in exact_candidates:
        if p and os.path.exists(p): return p
    # Deep search
    for root, dirs, files in os.walk("../input"):
        if filename in files:
            if search_hint and search_hint in root: return os.path.join(root, filename)
            return os.path.join(root, filename)
    return None

def find_data_root():
    candidates = ["../input/semevaltask9-testphase-data/semeval_polar_testphase", "../input/semeval_polar_testphase"]
    for c in candidates:
        if os.path.exists(c): return c
    for root, dirs, files in os.walk("../input"):
        if "subtask1" in dirs and "train" in dirs: return root
    return candidates[0]

Config.DATA_ROOT = find_data_root()


def load_data(mode="train"):
    print(f">>> Loading {mode.upper()} Data...")
    dfs = []
    search_path = os.path.join(Config.DATA_ROOT, f"subtask1/{mode}")
    if not os.path.exists(search_path): search_path = os.path.join(Config.DATA_ROOT, f"subtask_1/{mode}")
    if not os.path.exists(search_path): search_path = os.path.join(Config.DATA_ROOT, f"subtask2/{mode}")
    
    files = glob.glob(os.path.join(search_path, "*.csv"))
    if not files: raise ValueError("No CSVs found.")
    
    for f in files:
        df = pd.read_csv(f)
        df['lang'] = os.path.basename(f).split('_')[0].split('.')[0]
        dfs.append(df)
    master = pd.concat(dfs, ignore_index=True)
    
    if mode == "train":
        def read_sub(t):
            p = os.path.join(Config.DATA_ROOT, f"subtask{t}", "train")
            if not os.path.exists(p): p = os.path.join(Config.DATA_ROOT, f"subtask_{t}", "train")
            fs = glob.glob(os.path.join(p, "*.csv"))
            return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
        df2, df3 = read_sub(2), read_sub(3)
        if not df2.empty: master = pd.merge(master, df2[['id'] + Config.LABELS_S2], on='id', how='left')
        if not df3.empty: master = pd.merge(master, df3[['id'] + Config.LABELS_S3], on='id', how='left')
        if 'label' in master.columns: master.rename(columns={'label': 'polarization'}, inplace=True)
        master = master.fillna(0)
    return master

class PolarDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df; self.tokenizer = tokenizer
        self.text = df['text'].values if 'text' in df.columns else df['content'].values
        self.langs = df['lang'].values
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        text = f"<{self.langs[idx]}> {self.text[idx]}"
        enc = self.tokenizer(text, truncation=True, max_length=Config.MAX_LEN, padding=False)
        return {k: torch.tensor(v) for k,v in enc.items()}



MODEL_MAP = {
    "mdeberta_s1": { "hf": "microsoft/mdeberta-v3-base", "file": "best_microsoft_mdeberta-v3-base.pth", "type": "benchmark", "tasks": ["s1"] },
    "mdeberta_s23": { "hf": "microsoft/mdeberta-v3-base", "file": "best_model.pth", "hint": "task9_semeval3", "type": "grandmaster", "tasks": ["s2", "s3"] },
    "xlm-roberta": { "hf": "xlm-roberta-base", "file": "best_xlm-roberta.pth", "type": "grandmaster", "tasks": ["s1", "s2", "s3"] },
    "labse": { "hf": "sentence-transformers/LaBSE", "file": "best_labse.pth", "type": "grandmaster", "tasks": ["s1", "s2", "s3"] },
    "mmbert": { "hf": "jhu-clsp/mmBERT-base", "file": "best_mmbert.pth", "type": "grandmaster", "tasks": ["s1", "s2", "s3"] },
    "rembert": { "hf": "google/rembert", "file": "best_rembert.pth", "type": "grandmaster", "tasks": ["s1", "s2", "s3"] }
}

def safe_forward(backbone, config, input_ids, attention_mask, token_type_ids=None):
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if token_type_ids is not None:
        model_type = getattr(config, "model_type", "").lower()
        if not any(x in model_type for x in ["modern", "roberta", "rembert", "distilbert"]):
            inputs["token_type_ids"] = token_type_ids
    return backbone(**inputs)

class BenchmarkModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config, trust_remote_code=True)
        self.fc = nn.Linear(self.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        out = safe_forward(self.backbone, self.config, input_ids, attention_mask, token_type_ids)
        mask = attention_mask.unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        emb = torch.sum(out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return {"logits_s1": self.fc(emb)}

class GrandmasterModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
   
        self.config.output_hidden_states = True 
        
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config, trust_remote_code=True)
        self.pooler_weights = nn.Parameter(torch.tensor([1.0]*5)) 
        self.drop = nn.Dropout(0.1)
        self.h1 = nn.Linear(self.config.hidden_size, 1)
        self.h2 = nn.Linear(self.config.hidden_size, 5)
        self.h3 = nn.Linear(self.config.hidden_size + 6, 6)
    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        out = safe_forward(self.backbone, self.config, input_ids, attention_mask, token_type_ids)
        
      
        states = torch.stack(list(out.hidden_states)[-5:], 0)
        feat = self.drop((nn.functional.softmax(self.pooler_weights, 0).view(-1,1,1,1) * states).sum(0)[:, 0])
        o1 = self.h1(feat); o2 = self.h2(feat)
        o3 = self.h3(torch.cat([feat, o1.detach(), o2.detach()], 1))
        return {"logits_s1": o1, "logits_s2": o2, "logits_s3": o3}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_raw_predictions(df):
    raw_preds = {}
    print(f"   Generating predictions for {len(df)} samples...")
    
    for key, info in MODEL_MAP.items():
        path = find_model_path(info['file'], info.get('hint'))
        if not path:
            print(f"      ⚠️ {key} not found. Skipping.")
            continue
            
        print(f"      Processing {key}...")
        tokenizer = AutoTokenizer.from_pretrained(info['hf'], trust_remote_code=True)
        if "rembert" in info['hf']: tokenizer.model_max_length = Config.MAX_LEN
        dl = DataLoader(PolarDataset(df, tokenizer), batch_size=Config.BATCH_SIZE, shuffle=False, 
                        collate_fn=DataCollatorWithPadding(tokenizer), num_workers=2)
        
        model = BenchmarkModel(info['hf']) if info['type'] == 'benchmark' else GrandmasterModel(info['hf'])
        try:
         
            st = torch.load(path, map_location=device, weights_only=False)
            if 'model' in st: st = st['model']
            st = {k.replace('module.', '').replace('model.', 'backbone.'): v for k,v in st.items()}
            model.load_state_dict(st, strict=False)
        except Exception as e: 
            print(f"      ❌ Error loading {key}: {e}")
            continue
        
        model.to(device).eval(); p1, p2, p3 = [], [], []
        with torch.no_grad():
            for b in tqdm(dl, leave=False):
                b = {k: v.to(device) for k,v in b.items()}
                out = model(**b)
                p1.append(torch.sigmoid(out['logits_s1']).cpu().numpy())
                if info['type'] == 'grandmaster':
                    p2.append(torch.sigmoid(out['logits_s2']).cpu().numpy())
                    p3.append(torch.sigmoid(out['logits_s3']).cpu().numpy())
        
        del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
        
        if p1: p1 = np.concatenate(p1)
        if p2: p2 = np.concatenate(p2)
        if p3: p3 = np.concatenate(p3)
        raw_preds[key] = (p1, p2, p3)
        
    return raw_preds



def run_strategy(name, raw_preds, val_df, test_df_ids, test_df_langs):
    print(f"\n🧪 [STRATEGY: {name.upper()}] Running...")
    
    def aggregate(preds_dict, task, size):
        if name == "10_Rank_Average":
            accum = np.zeros(size)
            cnt = 0
            for k, v in preds_dict.items():
                if task in MODEL_MAP[k]['tasks']:
                    p = v[0] if task=='s1' else (v[1] if task=='s2' else v[2])
                    if len(p) > 0:
                        r = rankdata(p, axis=0) / len(p)
                        accum += r; cnt += 1
            return accum / cnt if cnt > 0 else accum

        elif name == "08_Power_Mean":
            accum = np.zeros(size)
            cnt = 0
            for k, v in preds_dict.items():
                if task in MODEL_MAP[k]['tasks']:
                    p = v[0] if task=='s1' else (v[1] if task=='s2' else v[2])
                    if len(p) > 0:
                        accum += p**2; cnt += 1
            return np.sqrt(accum / cnt) if cnt > 0 else accum
            
        elif name == "09_Max_Recall":
            accum = np.zeros(size)
            for k, v in preds_dict.items():
                if task in MODEL_MAP[k]['tasks']:
                    p = v[0] if task=='s1' else (v[1] if task=='s2' else v[2])
                    if len(p) > 0:
                        accum = np.maximum(accum, p)
            return accum

        else:
            accum = np.zeros(size)
            cnt = 0
            for k, v in preds_dict.items():
                if task in MODEL_MAP[k]['tasks']:
                    p = v[0] if task=='s1' else (v[1] if task=='s2' else v[2])
                    if len(p) > 0:
                        weight = 1.0
                        if name == "06_Weighted_Specialist" and k == "mdeberta_s1" and task == "s1": weight = 2.0
                        if name == "07_Weighted_Polyglot" and k in ["rembert", "mmbert"] and task in ["s2", "s3"]: weight = 1.5
                        accum += p * weight; cnt += weight
            return accum / cnt if cnt > 0 else accum

    N_val = len(val_df)
    vp1 = aggregate(raw_preds['val'], 's1', (N_val, 1))
    vp2 = aggregate(raw_preds['val'], 's2', (N_val, 5))
    vp3 = aggregate(raw_preds['val'], 's3', (N_val, 6))

    y1, y2, y3 = val_df['polarization'].values, val_df[Config.LABELS_S2].values, val_df[Config.LABELS_S3].values
    vp1_c, vp2_c, vp3_c = np.clip(vp1, 1e-7, 1-1e-7), np.clip(vp2, 1e-7, 1-1e-7), np.clip(vp3, 1e-7, 1-1e-7)
    
    
    th1, th2, th3 = 0.5, 0.35, 0.35 
    if "Optimized" in name or "SOTA" in name or "Strict" in name or "Rescue" in name:
        best_f, best_t = 0, 0.5
        for t in np.arange(0.3, 0.7, 0.02):
            if f1_score(y1, (vp1>t).astype(int), average='macro') > best_f: best_f, best_t = f1_score(y1, (vp1>t).astype(int), average='macro'), t
        th1 = best_t

    def post_proc(p1, p2, p3):
        pred1 = (p1 > th1).astype(int).flatten()
        pred2 = (p2 > th2).astype(int)
        pred3 = (p3 > th3).astype(int)
        
        if "Strict" in name or "Combo" in name or "SOTA" in name:
            pred2[pred1==0] = 0; pred3[pred1==0] = 0
            
        if "Rescue" in name or "Combo" in name or "SOTA" in name:
            mask = (pred1==1) & (pred2.sum(1)==0)
            if mask.sum()>0: pred2[mask, p2[mask].argmax(1)] = 1
            mask = (pred1==1) & (pred3.sum(1)==0)
            if mask.sum()>0: pred3[mask, p3[mask].argmax(1)] = 1
        return pred1, pred2, pred3

    val_pred1, val_pred2, val_pred3 = post_proc(vp1, vp2, vp3)

    print(f"   📊 [LOGS] {name}")
    print(f"      Loss | S1: {log_loss(y1, vp1_c):.4f} | S2: {log_loss(y2.flatten(), vp2_c.flatten()):.4f} | S3: {log_loss(y3.flatten(), vp3_c.flatten()):.4f}")
    print(f"      F1   | S1: {f1_score(y1, val_pred1, average='macro'):.4f} | S2: {f1_score(y2, val_pred2, average='macro'):.4f} | S3: {f1_score(y3, val_pred3, average='macro'):.4f}")
    
    N_test = len(test_df_ids)
    tp1 = aggregate(raw_preds['test'], 's1', (N_test, 1))
    tp2 = aggregate(raw_preds['test'], 's2', (N_test, 5))
    tp3 = aggregate(raw_preds['test'], 's3', (N_test, 6))
    
    fin1, fin2, fin3 = post_proc(tp1, tp2, tp3)
    
    base_dir = os.path.join(Config.OUTPUT_ROOT, name)
    os.makedirs(base_dir, exist_ok=True)
    langs = np.unique(test_df_langs)
    for lang in langs:
        idx = np.where(test_df_langs == lang)[0]; ids = test_df_ids[idx]
        
        os.makedirs(f"{base_dir}/subtask_1", exist_ok=True)
        pd.DataFrame({'id': ids, 'polarization': fin1[idx]}).to_csv(f"{base_dir}/subtask_1/pred_{lang}.csv", index=False)
        os.makedirs(f"{base_dir}/subtask_2", exist_ok=True)
        d = pd.DataFrame(fin2[idx], columns=Config.LABELS_S2); d.insert(0,'id',ids); d.to_csv(f"{base_dir}/subtask_2/pred_{lang}.csv", index=False)
        os.makedirs(f"{base_dir}/subtask_3", exist_ok=True)
        d = pd.DataFrame(fin3[idx], columns=Config.LABELS_S3); d.insert(0,'id',ids); d.to_csv(f"{base_dir}/subtask_3/pred_{lang}.csv", index=False)
        
    shutil.make_archive(f"{base_dir}_subtask1", 'zip', f"{base_dir}/subtask_1")
    shutil.make_archive(f"{base_dir}_subtask2", 'zip', f"{base_dir}/subtask_2")
    shutil.make_archive(f"{base_dir}_subtask3", 'zip', f"{base_dir}/subtask_3")
    print(f"      ✅ Zips created.")

if __name__ == "__main__":
    if os.path.exists(Config.OUTPUT_ROOT): shutil.rmtree(Config.OUTPUT_ROOT)
    os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)
    
    full_train = load_data("train")
    _, val_df = train_test_split(full_train, test_size=Config.VAL_SPLIT, stratify=full_train['lang'], random_state=Config.SEED)
    test_df = load_data("test")
    
    print("\n🚀 STEP 1: Generating Raw Predictions (One-time pass)...")
    cache = {
        'val': generate_raw_predictions(val_df),
        'test': generate_raw_predictions(test_df)
    }
    
    print("\n🚀 STEP 2: Running 10 Inference Strategies...")
    strategies = ["01_Baseline_Fixed", "02_Optimized_Thresh", "03_Strict_Hierarchy", "04_Rescue_Mode", "05_SOTA_Combo", "06_Weighted_Specialist", "07_Weighted_Polyglot", "08_Power_Mean", "09_Max_Recall", "10_Rank_Average"]
    
    for strat in strategies:
        run_strategy(strat, cache, val_df, test_df['id'].values, test_df['lang'].values)
        
    print("\n🏁 ALL 10 TECHNIQUES COMPLETE. DOWNLOAD ZIPS FROM OUTPUT.")