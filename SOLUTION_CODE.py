import os, sys, json, time, math, random, shutil, glob, traceback, subprocess
from collections import defaultdict, OrderedDict
from functools import partial
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler, Sampler

from tqdm import tqdm


# ============================================================================
# 0. DISCOVER WAREHOUSE_SORT + MANI_SKILL
# ============================================================================

print("=" * 70)
print("  MARSO HACK BERLIN 2026 — v5 Setup")
print("=" * 70)

HAVE_MANISKILL = False
HAVE_WAREHOUSE_SORT = False

# --- mani_skill ---
try:
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    HAVE_MANISKILL = True
    print("  [mani_skill] imported")
except ImportError:
    print("  [mani_skill] not found, installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mani_skill"])
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
        HAVE_MANISKILL = True
        print("  [mani_skill] installed + imported")
    except Exception as e:
        print(f"  [mani_skill] FAILED: {e}")

# --- warehouse_sort: scan known Kaggle paths ---
_REPO_SEARCH_PATHS = [
    "/kaggle/input/datasets/harshrajbhar/forked-from-marso-roboticsberlin-marso-hackathon/berlin-marso-hackathon-main",
    "/kaggle/input/berlin-marso-hackathon",
    "/kaggle/input/berlin-marso-hackathon-main",
]

# Also scan all /kaggle/input/ subdirs
if os.path.isdir("/kaggle/input"):
    for entry in os.listdir("/kaggle/input"):
        entry_path = os.path.join("/kaggle/input", entry)
        if not os.path.isdir(entry_path):
            continue
        for sub in os.listdir(entry_path):
            sub_path = os.path.join(entry_path, sub)
            if os.path.isdir(sub_path):
                if os.path.isdir(os.path.join(sub_path, "warehouse_sort")):
                    if sub_path not in _REPO_SEARCH_PATHS:
                        _REPO_SEARCH_PATHS.append(sub_path)
                # One more level deep (the dataset has berlin-marso-hackathon-main/ inside)
                for sub2 in os.listdir(sub_path):
                    sub2_path = os.path.join(sub_path, sub2)
                    if os.path.isdir(sub2_path) and os.path.isdir(os.path.join(sub2_path, "warehouse_sort")):
                        if sub2_path not in _REPO_SEARCH_PATHS:
                            _REPO_SEARCH_PATHS.append(sub2_path)

REPO_PATH = None
for p in _REPO_SEARCH_PATHS:
    ws_dir = os.path.join(p, "warehouse_sort")
    if os.path.isdir(ws_dir) and os.path.isfile(os.path.join(ws_dir, "__init__.py")):
        REPO_PATH = p
        if p not in sys.path:
            sys.path.insert(0, p)
        try:
            import warehouse_sort
            HAVE_WAREHOUSE_SORT = True
            print(f"  [warehouse_sort] found + imported from: {p}")
            break
        except ImportError as e:
            # Try adding parent dir
            parent = os.path.dirname(p)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            try:
                import warehouse_sort
                HAVE_WAREHOUSE_SORT = True
                print(f"  [warehouse_sort] imported from parent: {parent}")
                break
            except ImportError:
                pass

if not HAVE_WAREHOUSE_SORT:
    print("  [warehouse_sort] NOT FOUND in /kaggle/input/")
    print("  Searching for warehouse_sort in all subdirs...")
    # Last resort: recursive search
    for root, dirs, files in os.walk("/kaggle/input"):
        if "warehouse_sort" in dirs and os.path.isfile(os.path.join(root, "warehouse_sort", "__init__.py")):
            parent = root
            if parent not in sys.path:
                sys.path.insert(0, parent)
            try:
                import warehouse_sort
                HAVE_WAREHOUSE_SORT = True
                REPO_PATH = parent
                print(f"  [warehouse_sort] found via deep search: {parent}")
                break
            except ImportError:
                pass

CAN_EVAL = HAVE_MANISKILL and HAVE_WAREHOUSE_SORT
print(f"\n  CAN_EVAL: {CAN_EVAL}  (mani_skill={HAVE_MANISKILL}, warehouse_sort={HAVE_WAREHOUSE_SORT})")
if REPO_PATH:
    print(f"  REPO_PATH: {REPO_PATH}")
print()

# --- Required imports ---
try:
    import h5py
except ImportError:
    print("ERROR: h5py not installed"); sys.exit(1)
try:
    from diffusers.optimization import get_scheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.training_utils import EMAModel
except ImportError:
    print("ERROR: diffusers not installed"); sys.exit(1)
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("ERROR: gymnasium not installed"); sys.exit(1)

# Import warehouse_sort utils if available (for exact eval matching)
HAVE_WS_UTILS = False
if HAVE_WAREHOUSE_SORT:
    try:
        from warehouse_sort.utils import make_env, rollout_metrics, to_device, load_agent, expand_seeds
        from omegaconf import OmegaConf
        HAVE_WS_UTILS = True
        print("  [warehouse_sort.utils] imported (will use official eval pipeline)")
    except ImportError as e:
        print(f"  [warehouse_sort.utils] import failed: {e}")


# ============================================================================
# CONFIG
# ============================================================================

# Data paths: use the exact competition path
KAGGLE_DATA_DIR = "/kaggle/input/competitions/marso-hack-berlin-2026-robot-parcel-sorting-challenge"
if not os.path.isdir(KAGGLE_DATA_DIR):
    KAGGLE_DATA_DIR = "/kaggle/input/marso-hack"

OUTPUT_DIR = "/kaggle/working/marso_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED              = 42
TOTAL_ITERS       = 200000
BATCH_SIZE        = 64
GRAD_ACCUM_STEPS  = 4
LR                = 3e-4
WARMUP_STEPS      = 500
WEIGHT_DECAY      = 1e-6
EVAL_FREQ         = 5000
NUM_EVAL_EPISODES = 8
NUM_EVAL_ENVS     = 4
FINAL_EVAL_EPS    = 16
MAX_EPISODE_STEPS = 200

OBS_HORIZON       = 2
ACT_HORIZON       = 8
PRED_HORIZON      = 16
NUM_KP            = 32
UNET_DIMS         = [128, 256, 512]
DIFF_EMBED_DIM    = 64
N_GROUPS          = 8
NUM_INF_STEPS     = 16
USE_AUGMENTATION  = True
EMA_POWER         = 0.75
MAX_GRAD_NORM     = 1.0
ACTION_CLIP       = 1.0
NORMALIZE_ACTIONS = False
NORMALIZE_STATES  = False
AMP_DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# 1. HDF5 Demo Loading
# ============================================================================

def load_content_from_h5_file(file):
    if isinstance(file, (h5py.File, h5py.Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, h5py.Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unsupported h5 type: {type(file)}")

def load_traj_hdf5(path, num_traj=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"  Loading HDF5: {path}")
            f = h5py.File(path, "r")
            keys = list(f.keys())
            if num_traj is not None:
                keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))[:num_traj]
            ret = {key: load_content_from_h5_file(f[key]) for key in keys}
            f.close()
            return ret
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt+1}: {e}"); time.sleep(1)
            else:
                raise

def load_demo_dataset(path, num_traj=None):
    raw = load_traj_hdf5(path, num_traj)
    return {"observations": [raw[idx]["obs"] for idx in raw],
            "actions": [raw[idx]["actions"] for idx in raw]}


# ============================================================================
# 2. Observation Processing
# ============================================================================

def build_state_obs_extractor():
    return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())

def convert_obs(obs, concat_fn, transpose_fn, state_obs_extractor, depth=False):
    img_dict = obs["sensor_data"]
    rgb = transpose_fn(concat_fn([v["rgb"] for v in img_dict.values()]))
    states_to_stack = state_obs_extractor(obs)
    for j in range(len(states_to_stack)):
        if states_to_stack[j].dtype == np.float64:
            states_to_stack[j] = states_to_stack[j].astype(np.float32)
    try: state = np.hstack(states_to_stack)
    except: state = np.column_stack(states_to_stack)
    out = {"state": state, "rgb": rgb}
    if depth:
        dep = transpose_fn(concat_fn([v["depth"] for v in img_dict.values()]))
        out["depth"] = dep.astype(np.float16) if isinstance(dep, torch.Tensor) else dep
    return out

def reorder_keys(d, ref_dict):
    out = {}
    for k, v in ref_dict.items():
        if isinstance(v, (dict, spaces.Dict)):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


# ============================================================================
# 3. Networks: SpatialSoftmax + ResNet18 + ConditionalUnet1D
# ============================================================================

class SpatialSoftmax(nn.Module):
    def __init__(self, in_channels, num_kp=32):
        super().__init__()
        self.num_kp = num_kp
        self.kp_conv = nn.Conv2d(in_channels, num_kp, kernel_size=1) if num_kp else None
        self.out_channels = num_kp if num_kp else in_channels
    def forward(self, feat):
        if self.kp_conv is not None: feat = self.kp_conv(feat)
        b, c, h, w = feat.shape
        ys, xs = torch.meshgrid(
            torch.linspace(-1,1,h,device=feat.device,dtype=feat.dtype),
            torch.linspace(-1,1,w,device=feat.device,dtype=feat.dtype), indexing="ij")
        xs, ys = xs.reshape(1,1,h*w), ys.reshape(1,1,h*w)
        attn = F.softmax(feat.reshape(b,c,h*w), dim=-1)
        return torch.stack([(attn*xs).sum(-1), (attn*ys).sum(-1)], -1).reshape(b,2*c)

def _bn_to_gn(module, ng=16):
    for n, c in module.named_children():
        if isinstance(c, nn.BatchNorm2d):
            g = ng if c.num_features % ng == 0 else 1
            setattr(module, n, nn.GroupNorm(g, c.num_features))
        else: _bn_to_gn(c, ng)

class ResNet18SpatialSoftmax(nn.Module):
    def __init__(self, in_ch=3, out_dim=256, num_kp=32, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18
        try: net = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        except: net = resnet18(weights=None)
        if in_ch != 3: net.conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        self.trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
        _bn_to_gn(self.trunk)
        self.spatial_softmax = SpatialSoftmax(256, num_kp)
        self.fc = nn.Sequential(nn.Linear(2*num_kp, out_dim), nn.ReLU())
    def forward(self, x):
        return self.fc(self.spatial_softmax(self.trunk(x)))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, x):
        hd = self.dim//2; dev = x.device
        emb = torch.exp(torch.arange(hd, device=dev) * -(math.log(10000)/(hd-1)))
        return torch.cat([(x[:,None]*emb[None,:]).sin(), (x[:,None]*emb[None,:]).cos()], -1)

class Downsample1d(nn.Module):
    def __init__(self, d): super().__init__(); self.conv=nn.Conv1d(d,d,3,2,1)
    def forward(self, x): return self.conv(x)
class Upsample1d(nn.Module):
    def __init__(self, d): super().__init__(); self.conv=nn.ConvTranspose1d(d,d,4,2,1)
    def forward(self, x): return self.conv(x)
class Conv1dBlock(nn.Module):
    def __init__(self, i, o, k, ng=8): super().__init__(); self.block=nn.Sequential(nn.Conv1d(i,o,k,padding=k//2),nn.GroupNorm(ng,o),nn.Mish())
    def forward(self, x): return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, ic, oc, cd, ks=3, ng=8):
        super().__init__()
        self.blocks=nn.ModuleList([Conv1dBlock(ic,oc,ks,ng),Conv1dBlock(oc,oc,ks,ng)])
        self.out_channels=oc
        self.cond_encoder=nn.Sequential(nn.Mish(),nn.Linear(cd,oc*2),nn.Unflatten(-1,(-1,1)))
        self.residual_conv=nn.Conv1d(ic,oc,1) if ic!=oc else nn.Identity()
    def forward(self, x, cond):
        out=self.blocks[0](x)
        e=self.cond_encoder(cond).reshape(cond.shape[0],2,self.out_channels,1)
        out=self.blocks[1](e[:,0]*out+e[:,1])
        return out+self.residual_conv(x)

class ConditionalUnet1D(nn.Module):
    def __init__(self, idim, gcd, dsed=64, down_dims=None, ks=5, ng=8):
        super().__init__()
        if down_dims is None: down_dims=[128,256,512]
        ad=[idim]+list(down_dims); sd=down_dims[0]; cd=dsed+gcd
        self.diffusion_step_encoder=nn.Sequential(SinusoidalPosEmb(dsed),nn.Linear(dsed,dsed*4),nn.Mish(),nn.Linear(dsed*4,dsed))
        md=ad[-1]
        self.mid_modules=nn.ModuleList([ConditionalResidualBlock1D(md,md,cd,ks,ng)]*2)
        io=list(zip(ad[:-1],ad[1:]))
        self.down_modules=nn.ModuleList([nn.ModuleList([ConditionalResidualBlock1D(di,do,cd,ks,ng),ConditionalResidualBlock1D(do,do,cd,ks,ng),Downsample1d(do) if i<len(io)-1 else nn.Identity()]) for i,(di,do) in enumerate(io)])
        self.up_modules=nn.ModuleList([nn.ModuleList([ConditionalResidualBlock1D(do*2,di,cd,ks,ng),ConditionalResidualBlock1D(di,di,cd,ks,ng),Upsample1d(di) if i<len(io)-1 else nn.Identity()]) for i,(di,do) in enumerate(reversed(io[1:]))])
        self.final_conv=nn.Sequential(Conv1dBlock(sd,sd,ks),nn.Conv1d(sd,idim,1))
        print(f"  ConditionalUnet1D params: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")
    def forward(self, sample, timestep, global_cond=None):
        sample=sample.moveaxis(-1,-2)
        if not torch.is_tensor(timestep): timestep=torch.tensor([timestep],dtype=torch.long,device=sample.device)
        elif len(timestep.shape)==0: timestep=timestep[None].to(sample.device)
        timestep=timestep.expand(sample.shape[0])
        gf=self.diffusion_step_encoder(timestep)
        if global_cond is not None: gf=torch.cat([gf,global_cond],-1)
        x,h=sample,[]
        for r,r2,d in self.down_modules: x=r2(r(x,gf),gf); h.append(x); x=d(x)
        for m in self.mid_modules: x=m(x,gf)
        for r,r2,u in self.up_modules: x=r2(r(torch.cat((x,h.pop()),1),gf),gf); x=u(x)
        return self.final_conv(x).moveaxis(-1,-2)


# ============================================================================
# 4. Data Augmentation
# ============================================================================

class RGBAugmentation(nn.Module):
    def __init__(self, br=0.15, co=0.15, sa=0.15, cp=4, rep=0.1, gn=0.02):
        super().__init__(); self.br=br; self.co=co; self.sa=sa; self.cp=cp; self.rep=rep; self.gn=gn
    def forward(self, img):
        if not self.training: return img
        B,C,H,W=img.shape
        if self.br>0: img*=(1+(torch.rand(B,1,1,1,device=img.device)*2-1)*self.br)
        if self.co>0:
            cf=1+(torch.rand(B,1,1,1,device=img.device)*2-1)*self.co
            img=(img-img.mean([2,3],keepdim=True))*cf+img.mean([2,3],keepdim=True)
        if self.sa>0:
            sf=1+(torch.rand(B,1,1,1,device=img.device)*2-1)*self.sa
            img=img*sf+img.mean(1,keepdim=True)*(1-sf)
        img=img.clamp(0,1)
        if self.cp>0:
            p=self.cp; img=F.pad(img,[p]*4,mode='replicate')
            oh=torch.randint(0,2*p+1,(B,),device=img.device)
            ow=torch.randint(0,2*p+1,(B,),device=img.device)
            img=torch.cat([img[i:i+1,:,oh[i]:oh[i]+H,ow[i]:ow[i]+W] for i in range(B)])
        if self.rep>0:
            for i in range(B):
                if torch.rand(1).item()<self.rep:
                    eh,ew=torch.randint(8,32,(1,)).item(),torch.randint(8,32,(1,)).item()
                    ey,ex=torch.randint(0,max(1,H-eh),(1,)).item(),torch.randint(0,max(1,W-ew),(1,)).item()
                    img[i,:,ey:ey+eh,ex:ex+ew]=torch.rand(C,eh,ew,device=img.device)
        if self.gn>0: img=(img+torch.randn_like(img)*self.gn).clamp(0,1)
        return img


# ============================================================================
# 5. Agent
# ============================================================================

class Agent(nn.Module):
    def __init__(self, state_dim, act_dim, obs_horizon=2, act_horizon=8, pred_horizon=16,
                 visual_encoder="resnet18", num_kp=32, unet_dims=None,
                 diffusion_step_embed_dim=64, n_groups=8, use_augmentation=True,
                 action_mean=None, action_std=None, state_mean=None, state_std=None):
        super().__init__()
        if unet_dims is None: unet_dims=[128,256,512]
        self.obs_horizon=obs_horizon; self.act_horizon=act_horizon; self.pred_horizon=pred_horizon
        self.act_dim=act_dim; self.visual_feature_dim=256
        if visual_encoder=="resnet18":
            self.visual_encoder=ResNet18SpatialSoftmax(3,256,num_kp=num_kp)
        else: raise ValueError(f"Unknown encoder: {visual_encoder}")
        self.aug=RGBAugmentation() if use_augmentation else None
        # Normalization
        am = torch.tensor(action_mean,dtype=torch.float32) if action_mean is not None else torch.zeros(act_dim)
        ast = torch.tensor(action_std,dtype=torch.float32) if action_std is not None else torch.ones(act_dim)
        sm = torch.tensor(state_mean,dtype=torch.float32) if state_mean is not None else torch.zeros(state_dim)
        sst = torch.tensor(state_std,dtype=torch.float32) if state_std is not None else torch.ones(state_dim)
        self.register_buffer("action_mean",am); self.register_buffer("action_std",ast.clamp(min=1e-6))
        self.register_buffer("state_mean",sm); self.register_buffer("state_std",sst.clamp(min=1e-6))
        gcd=obs_horizon*(256+state_dim)
        self.noise_pred_net=ConditionalUnet1D(act_dim,gcd,diffusion_step_embed_dim,unet_dims,ng=n_groups)
        self.num_diffusion_iters=100
        self.noise_scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon")

    def encode_obs(self, obs_seq, eval_mode=False):
        rgb=obs_seq["rgb"].float()/255.0; B=rgb.shape[0]
        img_seq=rgb.flatten(end_dim=1)
        if self.aug is not None and not eval_mode: img_seq=self.aug(img_seq)
        vf=self.visual_encoder(img_seq).reshape(B,self.obs_horizon,-1)
        state=self.normalize_state(obs_seq["state"])
        return torch.cat((vf,state),-1).flatten(start_dim=1)

    def compute_loss(self, obs_seq, action_seq):
        B=obs_seq["state"].shape[0]
        obs_cond=self.encode_obs(obs_seq,eval_mode=False)
        a_normed=self.normalize_action(action_seq)
        noise=torch.randn((B,self.pred_horizon,self.act_dim),device=action_seq.device)
        ts=torch.randint(0,self.noise_scheduler.config.num_train_timesteps,(B,),device=action_seq.device).long()
        noisy=self.noise_scheduler.add_noise(a_normed,noise,ts)
        return F.mse_loss(self.noise_pred_net(noisy,ts,global_cond=obs_cond),noise)

    @torch.no_grad()
    def get_action(self, obs_seq, num_inference_steps=50):
        obs_seq["rgb"]=obs_seq["rgb"].permute(0,1,4,2,3)
        B=obs_seq["state"].shape[0]
        obs_cond=self.encode_obs(obs_seq,eval_mode=True)
        self.noise_scheduler.set_timesteps(num_inference_steps)
        noisy=torch.randn((B,self.pred_horizon,self.act_dim),device=obs_seq["state"].device)
        for k in self.noise_scheduler.timesteps:
            noisy=self.noise_scheduler.step(model_output=self.noise_pred_net(noisy,k,global_cond=obs_cond),timestep=k,sample=noisy).prev_sample
        start=self.obs_horizon-1
        return self.denormalize_action(noisy[:,start:start+self.act_horizon]).clamp(-ACTION_CLIP,ACTION_CLIP)

    def normalize_action(self, a): return (a-self.action_mean)/self.action_std
    def denormalize_action(self, a): return a*self.action_std+self.action_mean
    def normalize_state(self, s): return (s-self.state_mean)/self.state_std


# ============================================================================
# 6. Dataset
# ============================================================================

class IterationBasedBatchSampler(Sampler):
    def __init__(self, bs, ni, si=0): self.bs=bs; self.ni=ni; self.si=si
    def __iter__(self):
        it=self.si
        while it<self.ni:
            if hasattr(self.bs.sampler,"set_epoch"):
                self.bs.sampler.set_epoch(it)
            for b in self.bs:
                yield b
                it += 1
                if it >= self.ni:
                    break
    def __len__(self):
        return self.ni - self.si

class MultiDifficultyDataset(Dataset):
    def __init__(self, data_paths, obs_horizon, pred_horizon, device, num_traj_per_level=None):
        self.obs_horizon=obs_horizon; self.pred_horizon=pred_horizon; self.device=device
        self.slices=[]; self.trajectories={"observations":[],"actions":[]}
        soe=build_state_obs_extractor()
        opf=partial(convert_obs,concat_fn=partial(np.concatenate,axis=-1),
                    transpose_fn=partial(np.transpose,axes=(0,3,1,2)),state_obs_extractor=soe,depth=False)
        total=0
        for diff,h5p in data_paths.items():
            print(f"  Loading {diff}: {h5p}")
            raw=load_demo_dataset(h5p,num_traj_per_level)
            jp=h5p.replace(".h5",".json"); obs_space=None
            if os.path.exists(jp):
                with open(jp) as f: di=json.load(f)
                try:
                    kw=di.get("env_info",{}).get("env_kwargs",{})
                    tmp=gym.make("WarehouseSort-v1",sim_backend="gpu",obs_mode="rgb",control_mode="pd_ee_delta_pos",**kw)
                    obs_space=tmp.observation_space; tmp.close()
                except: pass
            for ti in range(len(raw["actions"])):
                ot=raw["observations"][ti]; at=raw["actions"][ti]
                if obs_space is not None: ot=reorder_keys(ot,obs_space)
                ot=opf(ot)
                ot["rgb"]=torch.from_numpy(ot["rgb"]).to(device)
                ot["state"]=torch.from_numpy(ot["state"]).to(device).float()
                at=torch.Tensor(at).to(device)
                tidx=len(self.trajectories["observations"])
                self.trajectories["observations"].append(ot); self.trajectories["actions"].append(at)
                L=at.shape[0]; assert ot["state"].shape[0]==L+1
                pb=obs_horizon-1; pa=pred_horizon-obs_horizon
                for s in range(-pb,L-pred_horizon+pa): self.slices.append((tidx,s,s+pred_horizon))
                total+=1
        self.pad_action_arm=torch.zeros((self.trajectories["actions"][0].shape[1]-1,),device=device)
        print(f"  Total trajectories: {total}  |  Total slices: {len(self.slices)}")
        self._compute_norm_stats()

    def _compute_norm_stats(self):
        print("  Computing normalization stats...")
        aa=torch.cat(self.trajectories["actions"]); ss=torch.cat([o["state"] for o in self.trajectories["observations"]])
        self.action_mean=aa.mean(0).cpu().numpy(); self.action_std=aa.std(0).cpu().numpy()
        self.state_mean=ss.mean(0).cpu().numpy(); self.state_std=ss.std(0).cpu().numpy()
        self.action_std=np.maximum(self.action_std,1e-6); self.state_std=np.maximum(self.state_std,1e-6)
        print(f"    action: mean={self.action_mean[:3]}... std={self.action_std[:3]}...")
        print(f"    state:  mean={self.state_mean[:3]}... std={self.state_std[:3]}...")

    def __getitem__(self, idx):
        tidx,s,e=self.slices[idx]; L=self.trajectories["actions"][tidx].shape[0]
        ot=self.trajectories["observations"][tidx]; obs={}
        for k,v in ot.items():
            obs[k]=v[max(0,s):s+self.obs_horizon]
            if s<0: obs[k]=torch.cat((torch.stack([obs[k][0]]*abs(s)),obs[k]),0)
        act=self.trajectories["actions"][tidx][max(0,s):e]
        if s<0: act=torch.cat([act[0].repeat(-s,1),act],0)
        if e>L:
            g=act[-1,-1]; pad=torch.cat((self.pad_action_arm,g[None]),0)
            act=torch.cat([act,pad.repeat(e-L,1)],0)
        return {"observations":obs,"actions":act}

    def __len__(self): return len(self.slices)


# ============================================================================
# 7. Evaluation (using repo's official utils when available)
# ============================================================================

# Difficulty configs matching the repo's conf/difficulty/*.yaml
DIFF_CONFIGS = {
    "easy":   {"num_parcels":2, "fixed_poses":True,
               "randomization":{"parcel_pose":{"xy_jitter":[0,0],"yaw_jitter":[0,0]},
                                "bin_position":{"side_swap_prob":0,"xy_jitter":[0,0]}}},
    "medium": {"num_parcels":4, "fixed_poses":False,
               "randomization":{"parcel_pose":{"xy_jitter":[-0.015,0.015],"yaw_jitter":[0,0]},
                                "bin_position":{"side_swap_prob":0,"xy_jitter":[0,0]}}},
    "hard":   {"num_parcels":6, "fixed_poses":False,
               "randomization":{"parcel_pose":{"xy_jitter":[-0.02,0.02],"yaw_jitter":[-0.1,0.1]},
                                "bin_position":{"side_swap_prob":0.5,"xy_jitter":[0,0]}}},
}

class EvalPolicy:
    """Policy wrapper with .act() compatible with warehouse_sort.utils.rollout_metrics.
    
    Implements proper ACTION CHUNKING: diffusion runs once, produces act_horizon actions,
    which are executed one-by-one. Only re-runs diffusion after all actions are consumed.
    This matches how the official train_rgbd evaluate.py works.
    """
    def __init__(self, agent, obs_horizon, act_horizon, device, num_inference_steps=50):
        self.agent=agent.to(device).eval()
        self.agent.noise_scheduler.set_timesteps(num_inference_steps)
        self.obs_horizon=obs_horizon; self.act_horizon=act_horizon
        self.device=device; self.prev=None
        self.action_queue=[]  # list of (B, act_dim) tensors, consumed one per step
        self._step_count=0; self._diffusion_count=0
        self._first_action_range=None

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        self._step_count += 1
        # If we still have buffered actions from the last diffusion run, use them
        if len(self.action_queue) > 0:
            action = self.action_queue.pop(0)
            self.prev = {"state": obs["state"].float().to(self.device), "rgb": obs["rgb"].to(self.device)}
            return action.clamp(-ACTION_CLIP, ACTION_CLIP)

        # Otherwise, run diffusion to get a new chunk of actions
        self._diffusion_count += 1
        state=obs["state"].float().to(self.device); rgb=obs["rgb"].to(self.device)
        
        # Debug: print eval obs shapes on first call
        if self._diffusion_count == 1:
            print(f"      [Eval] state shape={state.shape}, rgb shape={rgb.shape}, dtype={rgb.dtype}")
            print(f"      [Eval] state range=[{state.min():.3f}, {state.max():.3f}]")
            print(f"      [Eval] rgb range=[{rgb.min()}, {rgb.max()}]")
        
        cur={"state":state,"rgb":rgb}
        if self.prev is None or self.prev["state"].shape!=state.shape: self.prev=cur
        obs_seq={"state":torch.stack([self.prev["state"],state],1),"rgb":torch.stack([self.prev["rgb"],rgb],1)}
        self.prev=cur
        aseq=self.agent.get_action(obs_seq,num_inference_steps=self.agent.noise_scheduler.num_inference_steps)
        
        # Debug: print action stats on first call
        if self._first_action_range is None:
            self._first_action_range = (aseq.min().item(), aseq.max().item())
            print(f"      [Eval] action_seq shape={aseq.shape}, range=[{aseq.min():.4f}, {aseq.max():.4f}]")
            print(f"      [Eval] action[0]={aseq[0,0].cpu().numpy()}")
        
        # aseq shape: (B, act_horizon, act_dim)
        # Queue up actions 1..act_horizon-1, return action 0 immediately
        for i in range(1, aseq.shape[1]):
            self.action_queue.append(aseq[:, i])
        return aseq[:, 0].clamp(-ACTION_CLIP, ACTION_CLIP)


def _make_eval_env_manual(difficulty, num_envs=4):
    """Create eval env without repo utils (fallback)."""
    c=DIFF_CONFIGS[difficulty]; r=c["randomization"]
    env=gym.make("WarehouseSort-v1",num_envs=num_envs,obs_mode="rgb",
        control_mode="pd_ee_delta_pos",sim_backend="gpu",render_mode="rgb_array",
        reward_mode="sparse",max_episode_steps=MAX_EPISODE_STEPS,
        difficulty=difficulty,num_parcels=c["num_parcels"],fixed_poses=c["fixed_poses"],
        camera_width=128,camera_height=128,randomization=r)
    env=FlattenRGBDObservationWrapper(env,rgb=True,depth=False,state=True)
    env=ManiSkillVectorEnv(env,num_envs=num_envs,ignore_terminations=True,record_metrics=True)
    return env


def run_eval(ema_agent, difficulty, n_episodes, device):
    """Run evaluation for one difficulty, return sort_accuracy."""
    pol=EvalPolicy(ema_agent,OBS_HORIZON,ACT_HORIZON,device,NUM_INF_STEPS)

    if HAVE_WS_UTILS:
        # Use the repo's official pipeline (matches eval.py exactly)
        from omegaconf import OmegaConf
        cfg_dict={
            "control_mode":"pd_ee_delta_pos","max_episode_steps":MAX_EPISODE_STEPS,
            "camera":{"width":128,"height":128},"obs_camera":"scene",
            "difficulty":{"name":difficulty,"num_parcels":DIFF_CONFIGS[difficulty]["num_parcels"],
                          "fixed_poses":DIFF_CONFIGS[difficulty]["fixed_poses"]},
            "randomization":DIFF_CONFIGS[difficulty]["randomization"],
            "num_envs":NUM_EVAL_ENVS,"seed":0
        }
        cfg=OmegaConf.create(cfg_dict)
        rand=OmegaConf.create(DIFF_CONFIGS[difficulty]["randomization"])
        env,_=make_env(cfg,"rgb",rand,num_envs=NUM_EVAL_ENVS)
        seeds=list(range(5000,5000+n_episodes))
        m=rollout_metrics(env,pol,device,n_episodes,seeds,MAX_EPISODE_STEPS)
        print(f"      [Eval] steps={pol._step_count}, diffusion_runs={pol._diffusion_count}, "
              f"action_range={pol._first_action_range}")
        env.close()
        return m["sort_accuracy"]
    else:
        # Fallback: manual rollout
        env=_make_eval_env_manual(difficulty,NUM_EVAL_ENVS)
        base=env.unwrapped; nb=base.num_envs
        seeds=list(range(5000,5000+n_episodes))
        ts=tp=0
        for start in range(0,n_episodes,nb):
            bs=seeds[start:start+nb]; take=len(bs)
            if take<nb: bs+=seeds[:nb-take]
            obs,_=env.reset(seed=bs)
            for _ in range(MAX_EPISODE_STEPS-1):
                obs,_,_,_,_=env.step(pol.act(obs))
                obs={k:v.to(device) for k,v in obs.items()} if isinstance(obs,dict) else obs.to(device)
            ev=base.evaluate(); sc=ev["success_count"][:take]
            ts+=sc.sum().item(); tp+=base.num_parcels*take
        env.close()
        return ts/max(tp,1)


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    device=torch.device(DEVICE)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic=True
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    ckpt_dir=os.path.join(OUTPUT_DIR,"checkpoints"); os.makedirs(ckpt_dir,exist_ok=True)

    print("\n"+"="*70)
    print("  MARSO HACK BERLIN 2026 — v5")
    print("="*70)
    print(f"  Device: {device}  |  Data: {KAGGLE_DATA_DIR}")
    print(f"  Iters: {TOTAL_ITERS}  |  BS: {BATCH_SIZE}  |  EffBS: {BATCH_SIZE*GRAD_ACCUM_STEPS}")
    print(f"  LR: {LR}  |  EvalFreq: {EVAL_FREQ}  |  InfSteps: {NUM_INF_STEPS}")
    print(f"  CAN_EVAL: {CAN_EVAL}  |  Using official utils: {HAVE_WS_UTILS}")
    if REPO_PATH: print(f"  REPO_PATH: {REPO_PATH}")
    print("="*70)

    # STEP 1: Find demos
    print("\n  STEP 1: Locating demos")
    data_paths={}
    for diff in ["easy","medium","hard"]:
        cands=[
            os.path.join(KAGGLE_DATA_DIR,diff,"trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5"),
            os.path.join(KAGGLE_DATA_DIR,diff+".h5"),
        ]
        found=False
        for c in cands:
            if os.path.exists(c): data_paths[diff]=c; found=True; break
        if not found:
            dd=os.path.join(KAGGLE_DATA_DIR,diff)
            if os.path.isdir(dd):
                h5s=[f for f in os.listdir(dd) if f.endswith(".h5")]
                if h5s: data_paths[diff]=os.path.join(dd,h5s[0])
        print(f"    {diff}: {data_paths.get(diff,'NOT FOUND')}")
    if not data_paths: print(f"ERROR: No demos in {KAGGLE_DATA_DIR}"); sys.exit(1)

    # STEP 2: Load dataset
    print("\n  STEP 2: Loading data")
    dataset=MultiDifficultyDataset(data_paths,OBS_HORIZON,PRED_HORIZON,device)
    sample=dataset[0]; state_dim=sample["observations"]["state"].shape[-1]; act_dim=sample["actions"].shape[-1]
    print(f"    state_dim={state_dim}  act_dim={act_dim}")

    sampler=RandomSampler(dataset,replacement=False)
    bs=BatchSampler(sampler,BATCH_SIZE,drop_last=True)
    bs=IterationBasedBatchSampler(bs,TOTAL_ITERS)
    dl=DataLoader(dataset,batch_sampler=bs,num_workers=0)

    # STEP 3: Build agent
    print("\n  STEP 3: Building agent")
    am=dataset.action_mean if NORMALIZE_ACTIONS else None
    ast=dataset.action_std if NORMALIZE_ACTIONS else None
    sm=dataset.state_mean if NORMALIZE_STATES else None
    sst=dataset.state_std if NORMALIZE_STATES else None

    agent=Agent(state_dim,act_dim,OBS_HORIZON,ACT_HORIZON,PRED_HORIZON,"resnet18",NUM_KP,
                UNET_DIMS,DIFF_EMBED_DIM,N_GROUPS,USE_AUGMENTATION,am,ast,sm,sst).to(device)
    print(f"    Total params: {sum(p.numel() for p in agent.parameters())/1e6:.2f}M")

    optimizer=optim.AdamW(agent.parameters(),lr=LR,betas=(0.95,0.999),weight_decay=WEIGHT_DECAY)
    lr_sched=get_scheduler("cosine",optimizer,num_warmup_steps=WARMUP_STEPS,num_training_steps=TOTAL_ITERS)
    ema=EMAModel(parameters=agent.parameters(),power=EMA_POWER)
    ema_agent=Agent(state_dim,act_dim,OBS_HORIZON,ACT_HORIZON,PRED_HORIZON,"resnet18",NUM_KP,
                    UNET_DIMS,DIFF_EMBED_DIM,N_GROUPS,False,am,ast,sm,sst).to(device)
    scaler=torch.amp.GradScaler(AMP_DEVICE)

    # Sanity check: verify get_action works with HWC eval-format input
    print("    Sanity check: testing get_action with HWC eval-format input...")
    try:
        agent.eval()
        fake_obs = {
            "state": torch.randn(2, OBS_HORIZON, state_dim, device=device),
            "rgb": torch.randint(0, 255, (2, OBS_HORIZON, 128, 128, 3), dtype=torch.uint8, device=device),
        }
        with torch.no_grad():
            test_act = agent.get_action(fake_obs, num_inference_steps=5)
        print(f"    get_action OK: output shape={test_act.shape}, range=[{test_act.min():.3f}, {test_act.max():.3f}]")
        agent.train()
    except Exception as e:
        print(f"    WARNING: get_action failed: {e}")
        print(f"    This likely means the RGB format is wrong - check permute in get_action")
        agent.train()

    # STEP 4: Train
    print("\n  STEP 4: Training")
    print(f"    NOTE: acc stays 0.000 until first eval at iter {EVAL_FREQ}")
    print(f"    First eval may still be 0% — the model needs 10K-50K+ iters to learn sorting")
    print(f"    The 'eval_in' counter shows iterations until next eval")
    best_sort_acc=0.0; best_loss=float("inf"); ema_loss=0.0; loss_history=[]
    best_ckpt_path=os.path.join(ckpt_dir,"best_loss.pt")

    agent.train(); pbar=tqdm(total=TOTAL_ITERS,desc="Training",ncols=140); optimizer.zero_grad()

    for it,db in enumerate(dl):
        with torch.amp.autocast(AMP_DEVICE):
            loss=agent.compute_loss(db["observations"],db["actions"])/GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()
        if (it+1)%GRAD_ACCUM_STEPS==0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(agent.parameters(),MAX_GRAD_NORM)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            lr_sched.step(); ema.step(agent.parameters())

        lv=loss.item()*GRAD_ACCUM_STEPS; loss_history.append(lv)
        ema_loss=lv if ema_loss==0 else 0.99*ema_loss+0.01*lv

        if it%200==0:
            next_eval=EVAL_FREQ-(it%EVAL_FREQ) if it>0 else EVAL_FREQ
            pbar.set_postfix({"loss":f"{lv:.5f}","ema":f"{ema_loss:.5f}",
                              "lr":f"{optimizer.param_groups[0]['lr']:.2e}",
                              "acc":f"{best_sort_acc:.3f}","eval_in":f"{next_eval}"})

        if ema_loss<best_loss and it>100 and it%500==0:
            best_loss=ema_loss; ema.copy_to(ema_agent.parameters())
            torch.save({"agent":agent.state_dict(),"ema_agent":ema_agent.state_dict(),"iteration":it,
                        "ema_loss":ema_loss,"action_mean":dataset.action_mean,"action_std":dataset.action_std,
                        "state_mean":dataset.state_mean,"state_std":dataset.state_std},best_ckpt_path)

        if it>0 and it%20000==0:
            ema.copy_to(ema_agent.parameters())
            p=os.path.join(ckpt_dir,f"iter_{it}.pt")
            torch.save({"agent":agent.state_dict(),"ema_agent":ema_agent.state_dict(),"iteration":it,
                        "ema_loss":ema_loss,"action_mean":dataset.action_mean,"action_std":dataset.action_std,
                        "state_mean":dataset.state_mean,"state_std":dataset.state_std},p)
            print(f"\n    Saved: {p}")

        if CAN_EVAL and it>0 and it%EVAL_FREQ==0:
            print(f"\n  -- Eval at iter {it} --")
            ema.copy_to(ema_agent.parameters()); was_train=agent.training; ema_agent.eval()
            try:
                scores={}
                for d in ["easy","medium","hard"]:
                    acc=run_eval(ema_agent,d,NUM_EVAL_EPISODES,device)
                    scores[d]=acc; print(f"    {d}: {acc:.4f}")
                W={"easy":0.2,"medium":0.3,"hard":0.5}
                ws=sum(W[d]*scores.get(d,0) for d in W); print(f"    Weighted: {ws:.4f}")
                if ws>best_sort_acc:
                    best_sort_acc=ws; ema.copy_to(ema_agent.parameters())
                    ap=os.path.join(ckpt_dir,"best_sort_accuracy.pt")
                    torch.save({"agent":agent.state_dict(),"ema_agent":ema_agent.state_dict(),"iteration":it,
                                "weighted_score":ws,"scores":scores,
                                "action_mean":dataset.action_mean,"action_std":dataset.action_std,
                                "state_mean":dataset.state_mean,"state_std":dataset.state_std},ap)
                    print(f"    *** New best: {ws:.4f}"); best_ckpt_path=ap
            except Exception as e:
                print(f"    Eval error: {e}"); traceback.print_exc()
            if was_train: agent.train()

        pbar.update(1)

    pbar.close(); ema.copy_to(ema_agent.parameters())

    # Save final
    fp=os.path.join(ckpt_dir,"final.pt")
    torch.save({"agent":agent.state_dict(),"ema_agent":ema_agent.state_dict(),"iteration":TOTAL_ITERS,
                "ema_loss":ema_loss,"weighted_score":best_sort_acc,
                "action_mean":dataset.action_mean,"action_std":dataset.action_std,
                "state_mean":dataset.state_mean,"state_std":dataset.state_std},fp)
    np.save(os.path.join(OUTPUT_DIR,"loss_history.npy"),np.array(loss_history))

    ap=os.path.join(ckpt_dir,"best_sort_accuracy.pt")
    if os.path.exists(ap) and best_sort_acc>0: best_ckpt_path=ap
    print(f"\n  Training done. Best acc: {best_sort_acc:.4f}  Best loss: {best_loss:.5f}")
    print(f"  Checkpoint: {best_ckpt_path}")

    # STEP 5: Final eval
    if CAN_EVAL:
        print("\n  STEP 5: Final Evaluation")
        ema_agent.eval(); scores={}
        for d in ["easy","medium","hard"]:
            try:
                acc=run_eval(ema_agent,d,FINAL_EVAL_EPS,device)
                scores[d]=acc; print(f"    {d}: {acc*100:.2f}%")
            except Exception as e:
                print(f"    {d}: error {e}"); scores[d]=0
        W={"easy":0.2,"medium":0.3,"hard":0.5}
        fs=sum(W[d]*scores.get(d,0) for d in W)
        print("\n"+"-"*50)
        for d in W: print(f"    {d:8s}: {scores.get(d,0)*100:6.2f}%  (w={W[d]})")
        print(f"    {'FINAL':8s}: {fs*100:6.2f}%")
        print("-"*50)
        with open(os.path.join(OUTPUT_DIR,"eval_scores.json"),"w") as f:
            json.dump({"scores":scores,"final_score":fs,"weights":W},f,indent=2)
    else:
        print("\n  STEP 5: Eval skipped (warehouse_sort not found)")

    # STEP 6: Submission files
    print("\n  STEP 6: Creating submission files")

    # Copy our Agent code + checkpoint into the repo for the judge
    if REPO_PATH:
        # Copy this script into the repo
        import shutil as sh
        script_src = os.path.abspath(__file__)
        script_dst = os.path.join(REPO_PATH, "solve_marso_hack_v5.py")
        try: sh.copy2(script_src, script_dst); print(f"    Copied script -> {script_dst}")
        except: pass

        # Copy checkpoint into the repo
        ckpt_dst_dir = os.path.join(REPO_PATH, "checkpoints")
        os.makedirs(ckpt_dst_dir, exist_ok=True)
        ckpt_dst = os.path.join(ckpt_dst_dir, "best.pt")
        try: sh.copy2(best_ckpt_path, ckpt_dst); print(f"    Copied checkpoint -> {ckpt_dst}")
        except: pass

        # Copy norm_stats
        try:
            with open(os.path.join(REPO_PATH, "norm_stats.json"), "w") as f:
                json.dump({"action_mean":dataset.action_mean.tolist(),"action_std":dataset.action_std.tolist(),
                           "state_mean":dataset.state_mean.tolist(),"state_std":dataset.state_std.tolist()},f,indent=2)
        except: pass

    # Write improved_policy.py into the repo (the policy entrypoint for eval.py)
    policy_dir = REPO_PATH if REPO_PATH else OUTPUT_DIR
    policy_path = os.path.join(policy_dir, "improved_policy.py")

    policy_code = '''"""Custom policy entrypoint for competition — v5 with normalization.

Usage with eval.py:
  python eval.py difficulty=easy obs_mode=rgb \\
      policy=improved_policy:load_improved_dp_rgb \\
      checkpoint=checkpoints/best.pt \\
      eval_config=conf/eval/default.yaml
"""
import torch, sys, os, json
import numpy as np

class ImprovedDPRgbPolicy:
    def __init__(self, agent, obs_horizon, act_horizon, device, num_inference_steps=''' + str(NUM_INF_STEPS) + '''):
        self.agent=agent.to(device).eval()
        self.agent.noise_scheduler.set_timesteps(num_inference_steps)
        self.obs_horizon=obs_horizon; self.act_horizon=act_horizon
        self.device=device; self.prev=None; self.action_queue=[]

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        if len(self.action_queue) > 0:
            action = self.action_queue.pop(0)
            self.prev = {"state": obs["state"].float().to(self.device), "rgb": obs["rgb"].to(self.device)}
            return action.clamp(-''' + str(ACTION_CLIP) + ''',''' + str(ACTION_CLIP) + ''')
        state=obs["state"].float().to(self.device); rgb=obs["rgb"].to(self.device)
        cur={"state":state,"rgb":rgb}
        if self.prev is None or self.prev["state"].shape!=state.shape: self.prev=cur
        obs_seq={"state":torch.stack([self.prev["state"],state],1),"rgb":torch.stack([self.prev["rgb"],rgb],1)}
        self.prev=cur
        aseq=self.agent.get_action(obs_seq,num_inference_steps=self.agent.noise_scheduler.num_inference_steps)
        for i in range(1, aseq.shape[1]):
            self.action_queue.append(aseq[:, i])
        return aseq[:, 0].clamp(-''' + str(ACTION_CLIP) + ''',''' + str(ACTION_CLIP) + ''')

def load_improved_dp_rgb(checkpoint, sample_obs, action_space, device):
    solve_path=os.path.dirname(os.path.abspath(__file__))
    if solve_path not in sys.path: sys.path.insert(0,solve_path)
    from solve_marso_hack_v5 import Agent

    state_dim=sample_obs["state"].shape[1]; act_dim=action_space.shape[0]
    ckpt=torch.load(checkpoint,map_location=device,weights_only=False)
    am=ckpt.get("action_mean",None); ast=ckpt.get("action_std",None)
    sm=ckpt.get("state_mean",None); sst=ckpt.get("state_std",None)
    if am is None:
        for ns_path in [os.path.join(os.path.dirname(checkpoint),"..","norm_stats.json"),
                        os.path.join(solve_path,"norm_stats.json")]:
            if os.path.exists(ns_path):
                with open(ns_path) as f: ns=json.load(f)
                am=np.array(ns["action_mean"]); ast=np.array(ns["action_std"])
                sm=np.array(ns["state_mean"]); sst=np.array(ns["state_std"]); break

    agent=Agent(state_dim,act_dim,''' + str(OBS_HORIZON) + ''',''' + str(ACT_HORIZON) + ''',''' + str(PRED_HORIZON) + ''',
        "resnet18",''' + str(NUM_KP) + ''',''' + str(UNET_DIMS) + ''',''' + str(DIFF_EMBED_DIM) + ''',''' + str(N_GROUPS) + ''',
        False,am,ast,sm,sst)
    agent.load_state_dict(ckpt.get("ema_agent",ckpt.get("agent")))
    return ImprovedDPRgbPolicy(agent,''' + str(OBS_HORIZON) + ''',''' + str(ACT_HORIZON) + ''',device)
'''
    with open(policy_path, "w") as f: f.write(policy_code)
    print(f"    improved_policy.py -> {policy_path}")

    # submission.yaml at repo root (matches competition format)
    sub_path = os.path.join(policy_dir, "submission.yaml")
    # Use relative paths if in repo, absolute if not
    if REPO_PATH:
        ckpt_rel = "checkpoints/best.pt"
    else:
        ckpt_rel = best_ckpt_path

    sub_yaml = f"""team: "marso-hack-team"
policy: improved_policy:load_improved_dp_rgb
obs_mode: rgb

levels:
  easy:   {{checkpoint: {ckpt_rel}}}
  medium: {{checkpoint: {ckpt_rel}}}
  hard:   {{checkpoint: {ckpt_rel}}}
"""
    with open(sub_path, "w") as f: f.write(sub_yaml)
    print(f"    submission.yaml -> {sub_path}")

    # Also save to OUTPUT_DIR
    with open(os.path.join(OUTPUT_DIR,"submission.yaml"),"w") as f: f.write(sub_yaml)
    with open(os.path.join(OUTPUT_DIR,"norm_stats.json"),"w") as f:
        json.dump({"action_mean":dataset.action_mean.tolist(),"action_std":dataset.action_std.tolist(),
                   "state_mean":dataset.state_mean.tolist(),"state_std":dataset.state_std.tolist()},f,indent=2)

    print("\n"+"="*70)
    print("  ALL DONE!")
    print("="*70)
    print()
    print("  FILES IN YOUR REPO:")
    if REPO_PATH:
        for f in ["solve_marso_hack_v5.py","improved_policy.py","submission.yaml","norm_stats.json","checkpoints/best.pt"]:
            fp=os.path.join(REPO_PATH,f)
            print(f"    {'OK' if os.path.exists(fp) else 'MISSING'}  {f}")
    print()
    print("  TO VERIFY (on this notebook):")
    if REPO_PATH:
        print(f"    cd {REPO_PATH}")
        print(f"    python eval.py difficulty=easy obs_mode=rgb \\")
        print(f"        policy=improved_policy:load_improved_dp_rgb \\")
        print(f"        checkpoint=checkpoints/best.pt \\")
        print(f"        eval_config=conf/eval/default.yaml")
    print()
    print("  TO SUBMIT:")
    print("    1. Push your repo to GitHub")
    print("    2. Submit on Kaggle Writeups with your repo URL")


if __name__=="__main__":
    main()
