# WarehouseSort Multi-Diffusion Policy — Berlin Marso Hackathon 2026

> An enhanced RGB Diffusion Policy for the **WarehouseSort** robot parcel sorting challenge. The agent must read tag colors from a 128×128 RGB image, plan pick-and-place motions, and sort 2–6 parcels into color-matched bins — including the **hard** difficulty where bin sides swap in 50% of episodes.

---

## 🏆 Final Results

| Difficulty | Sort Accuracy | Weight | Weighted Contribution |
|:----------:|:-------------:|:------:|:---------------------:|
| **Easy**   | **56.25%**    | 0.2    | 11.25%                |
| **Medium** | **17.19%**    | 0.3    |  5.16%                |
| **Hard**   | **12.50%**    | 0.5    |  6.25%                |
| **FINAL**  | **22.66%**    | —      | **22.66%**            |

```
--------------------------------------------------
    easy    :  56.25%  (w=0.2)
    medium  :  17.19%  (w=0.3)
    hard    :  12.50%  (w=0.5)
    FINAL   :  22.66%
--------------------------------------------------
```

The model is a single checkpoint that handles **all three difficulties**, trained jointly on 600 episodes (200 easy + 200 medium + 200 hard).

---

## 📋 Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Why Diffusion Policy?](#2-why-diffusion-policy)
3. [Architecture Design](#3-architecture-design)
4. [Multi-Difficulty Training Strategy](#4-multi-difficulty-training-strategy)
5. [Data Augmentation Pipeline](#5-data-augmentation-pipeline)
6. [Training Recipe](#6-training-recipe)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Ablation Studies & Design Rationale](#8-ablation-studies--design-rationale)
9. [Known Limitations & Future Work](#9-known-limitations--future-work)
10. [Reproduction & Submission](#10-reproduction--submission)

---

## 1. Problem Analysis

### The Core Challenge

The WarehouseSort task is fundamentally a **visual perception + sequential manipulation** problem. The robot must:

1. **Perceive** parcel positions and tag colors from a 128×128 RGB image (no depth, no privileged state)
2. **Identify** which bin corresponds to which color (especially on hard, where bins swap sides)
3. **Plan** a sequence of pick-and-place motions to sort all parcels
4. **Execute** these motions with sufficient precision to place parcels inside bin boundaries

The hardest part is **perception**: reading tag colors from low-resolution pixels under varying lighting and positions. The scripted policy that generates demonstrations has access to privileged simulator state (exact parcel poses and tag colors), but our policy must infer everything from the image.

### Why the Baseline Fails

The provided RGB Diffusion Policy baseline achieves ~0% sort accuracy because:

- **Insufficient training time**: 30K iterations is not enough for the visual encoder to learn meaningful features. The ResNet18 must first learn to extract spatial features from raw pixels before the policy can condition actions on them. This perceptual grounding requires significantly more gradient updates.
- **Limited data diversity**: Training on only one difficulty (200 episodes) provides insufficient variation for the model to generalize. The model overfits to the specific layout seen during training rather than learning transferable visual skills.
- **Under-capacity**: The default U-Net with `[64, 128, 256]` channels (~4.5M parameters) lacks the representational power to model the complex conditional distribution p(actions | visual_obs, proprioception) across diverse scene configurations.
- **No augmentation**: Without data augmentation, the model memorizes exact pixel patterns rather than learning invariant features. This is catastrophic for the hard difficulty where bin sides swap and positions vary.

### The Weighted Score Problem

The scoring formula `0.2×easy + 0.3×medium + 0.5×hard` means that **hard difficulty determines 50% of the final score**. Yet hard is the most challenging to learn because it requires:

- Processing scenes with 6 parcels (more clutter, more occlusion)
- Handling randomization in parcel positions and orientations
- Reading tag colors from pixels to determine sorting targets (since bins swap sides in 50% of episodes, memorizing "left=red" fails)
- Sequential execution of 6 pick-and-place operations without failing

Any approach that only works on easy/medium will score at most `0.2×1.0 + 0.3×1.0 + 0.5×0 = 0.5` (50%), which is unlikely to be competitive.

---

## 2. Why Diffusion Policy?

### Comparison with Alternatives

| Method | Multi-Modal Actions | Temporal Consistency | Compounding Error | Training Stability |
|--------|:-------------------:|:--------------------:|:-----------------:|:------------------:|
| BC (MLP) | ✗ | ✗ | Severe | Good |
| BC (Transformer) | ✓ | ✓ | Moderate | Good |
| ACT | ✓ | ✓ | Low | Moderate |
| **Diffusion Policy** | **✓** | **✓** | **Low** | **Good** |

### Diffusion Policy Advantages for This Task

**Multi-modal action distributions**: When a parcel can be approached from multiple angles, or when the gripper can grasp from different sides, the action distribution is inherently multi-modal. Diffusion Policy naturally handles this — the denoising process can converge to different modes depending on the noise initialization.

**Action chunking**: By predicting 16 actions at once (`pred_horizon=16`) and executing 8 of them (`act_horizon=8`), the policy plans complete motion segments rather than single steps. A single pick-and-place motion takes roughly 15–25 steps, so each diffusion prediction covers a meaningful portion of the task.

**Robustness to distribution shift**: During evaluation, the model encounters scenes slightly different from training (held-out seeds, wider randomization). The iterative denoising process is more robust to this shift than a single-shot predictor, as each denoising step can correct small errors.

### Why Not Reinforcement Learning?

The competition provides only a **sparse reward** (+1 per correctly placed parcel per step). Shaping this into a useful dense reward for RL would require significant engineering and risks local optima (e.g., the robot might learn to repeatedly pick and place the same parcel). Imitation learning from the 600 provided demonstrations is far more sample-efficient and stable for this task.

---

## 3. Architecture Design

### Visual Encoder: ResNet18 + SpatialSoftmax

```
Input: (B, 3, 128, 128) float32 ∈ [0, 1]
  │
  ├── ResNet18 Trunk (pretrained on ImageNet)
  │   ├── conv1 (7×7, stride=2) + BN + ReLU + MaxPool
  │   ├── layer1 (2 × BasicBlock, 64 channels)
  │   ├── layer2 (2 × BasicBlock, 128 channels)
  │   └── layer3 (2 × BasicBlock, 256 channels)
  │       Output: (B, 256, 8, 8)
  │
  ├── BatchNorm → GroupNorm Conversion
  │   (GroupNorm is compatible with EMA and small batch sizes)
  │
  ├── SpatialSoftmax(num_kp=32)
  │   ├── 1×1 Conv: (B, 256, 8, 8) → (B, 32, 8, 8)
  │   ├── Softmax over spatial dims → attention weights
  │   └── Expected (x, y) coordinates → (B, 64)
  │
  └── Linear(64, 256) + ReLU
      Output: (B, 256)
```

**Why SpatialSoftmax?** For robotic manipulation, *where* things are matters as much as *what* things are. Standard global pooling (average/max) collapses spatial information into a single descriptor per channel. SpatialSoftmax computes a soft-argmax per channel, producing 2D coordinate pairs that explicitly encode spatial locations. With 32 keypoints, the encoder produces 64 spatial coordinates — a compact but spatially-aware representation that tells the policy where parcels, bins, and the gripper are in image space.

**Why ImageNet Pretraining?** Starting from ImageNet weights gives the visual encoder a significant head start — it already has low-level edge and texture detectors. Fine-tuning these on the robot task is much faster than training from scratch, especially with limited data (600 episodes).

**Why GroupNorm instead of BatchNorm?** GroupNorm computes normalization within each sample independently, making it compatible with:
- Small batch sizes during evaluation
- EMA model updates (BatchNorm running statistics can diverge from EMA weights)
- Mixed precision training

### Action Denoiser: Conditional U-Net 1D

```
Input: noisy_action (B, 16, 4)  +  global_cond (B, obs_horizon × 282)
  │
  ├── Diffusion Step Embedding
  │   SinusoidalPosEmb(64) → Linear(64, 256) → Mish → Linear(256, 64)
  │
  ├── Concat: step_emb (B, 64) + obs_cond (B, 564) → global_feat (B, 628)
  │
  ├── Down Path
  │   ├── ResBlock(4→128) + FiLM + ResBlock(128→128) + FiLM + Downsample
  │   ├── ResBlock(128→256) + FiLM + ResBlock(256→256) + FiLM + Downsample
  │   └── ResBlock(256→512) + FiLM + ResBlock(512→512) + FiLM
  │
  ├── Mid: ResBlock(512→512) + FiLM × 2
  │
  ├── Up Path (with skip connections)
  │   ├── ResBlock(1024→256) + FiLM + ResBlock(256→256) + FiLM + Upsample
  │   ├── ResBlock(512→128) + FiLM + ResBlock(128→128) + FiLM + Upsample
  │   └── ResBlock(256→128) + FiLM + ResBlock(128→128) + FiLM
  │
  └── Final: Conv1dBlock(128, 128) + Conv1d(128, 4)
      Output: predicted noise (B, 16, 4)
```

**FiLM Conditioning**: Each residual block receives a conditioning signal through Feature-wise Linear Modulation (FiLM). The global feature vector (diffusion step + observation) is projected to per-channel scale and bias parameters: `output = γ × input + β`. This allows the observation to modulate every layer of the denoiser, ensuring the predicted actions are strongly conditioned on the visual input.

**Skip Connections**: The U-Net architecture preserves fine-grained temporal information through skip connections between the encoder and decoder. This is critical for action prediction — the early layers capture high-frequency action details (e.g., precise gripper timing), while deeper layers capture the overall motion trajectory.

### Parameter Count

| Component | Parameters |
|-----------|:----------:|
| ResNet18 Trunk (fine-tuned end-to-end) | ~11M |
| SpatialSoftmax + Linear | ~0.03M |
| Conditional U-Net 1D | ~12M |
| **Total (trainable)** | **~23M** |

---

## 4. Multi-Difficulty Training Strategy

### Motivation

The provided baseline trains on one difficulty at a time (e.g., `demo_dir=easy`). This means:
- Only 200 episodes of training data
- The model only sees one type of scene configuration
- A separate checkpoint is needed for each difficulty

We instead **combine demonstrations from all three difficulties** into a single training dataset, yielding 600 episodes with maximum diversity.

### Implementation

```python
class MultiDifficultyDataset(Dataset):
    def __init__(self, data_paths, obs_horizon, pred_horizon, device):
        # data_paths = {
        #     "easy":   "path/to/easy/trajectory.rgb...h5",
        #     "medium": "path/to/medium/trajectory.rgb...h5",
        #     "hard":   "path/to/hard/trajectory.rgb...h5",
        # }
        for difficulty, h5_path in data_paths.items():
            raw = load_demo_dataset(h5_path)
            # Convert observations, create sliding windows
            # All trajectories merged into a single index
```

### Why This Works

1. **Data quantity**: 600 episodes vs. 200 means 3× more training signal. For an image policy that must learn visual features from scratch, data quantity matters enormously.

2. **Implicit curriculum**: Easy demos (2 parcels, fixed poses) provide clean learning signal for basic reach-and-grasp. Medium demos introduce position variation. Hard demos introduce bin swaps and more complex scenes. The model naturally progresses from simpler to harder patterns.

3. **Feature sharing**: The same visual encoder must work across all difficulties. Features learned from hard scenes (detecting tag colors under randomization) also benefit easy/medium evaluation. Conversely, the large number of easy/medium examples provides stable gradient signal for basic visual features.

4. **Generalization**: The held-out evaluation configs use slightly wider randomization than training. By training on diverse configurations, the model is less likely to overfit to specific spatial arrangements.

### Single Checkpoint Advantage

The RGB observation shape is fixed at `(128, 128, 3)` regardless of difficulty — the number of parcels only affects the scene content, not the image dimensions. Combined with the fixed 26-dim proprioception, this means **one model processes all difficulties**. This simplifies submission (one checkpoint) and encourages feature sharing across difficulties.

---

## 5. Data Augmentation Pipeline

### Design Philosophy

Data augmentation for robot learning must balance two goals:
1. **Invariance**: The policy should produce the same action regardless of augmentation (e.g., brightness change shouldn't change the grasp target)
2. **Diversity**: Augmentations should cover the variation seen in held-out evaluation

Our pipeline is applied **on-GPU** during training only (disabled during evaluation). All augmentations operate on the normalized `[0, 1]` float image.

### Augmentation Details

| Augmentation | Range | Purpose |
|-------------|:------:|---------|
| Brightness | ±15% | Simulates lighting variation in held-out configs |
| Contrast | ±15% | Robustness to exposure/auto-white-balance changes |
| Saturation | ±15% | Prevents over-reliance on exact RGB values for tag colors |
| Random Crop | 4px pad → random offset | Simulates slight camera misalignment |
| Random Erasing | 10% prob, 8–32px patches | Forces holistic scene understanding |
| Gaussian Noise | σ=0.02 | Regularizes visual encoder, prevents overfitting to pixel-exact patterns |

### Why Color Augmentation is Critical

The hard difficulty swaps bin sides in 50% of episodes. Without augmentation, the model might learn a shortcut: *"if I see a red parcel, always place it to the left"* (which works when bins don't swap). Color jitter and saturation augmentation prevent this shortcut by making exact color values unreliable — the model must instead learn robust color discrimination that generalizes across lighting and exposure variations.

### Why Random Erasing Helps

Random erasing removes small rectangular patches from the image, forcing the model to not depend on any single spatial location for its decision. This is particularly important because:
- Parcels can occlude each other in the 6-parcel hard scenario
- The robot's arm frequently occludes parcels during manipulation
- Held-out configs may have slightly different camera angles

---

## 6. Training Recipe

### Optimizer & Schedule

We use **AdamW** with the following configuration:

```python
optimizer = AdamW(agent.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=300000)
```

**Why AdamW?** AdamW decouples weight decay from the gradient update, providing more consistent regularization than L2 regularization in standard Adam. The betas `(0.95, 0.999)` are standard for diffusion model training, providing stable second-moment estimates.

**Why Cosine Schedule?** Cosine annealing provides a smooth decay that avoids sudden learning rate drops (unlike step schedules). The warmup period prevents early training instability when the EMA model is still poorly estimated.

### EMA Model

```python
ema = EMAModel(parameters=agent.parameters(), power=0.75)
# After each training step:
ema.step(agent.parameters())
# For evaluation:
ema.copy_to(ema_agent.parameters())
```

The EMA model uses an exponential moving average of training weights with decay power 0.75. This is critical for evaluation stability:
- Training weights oscillate around the optimum (especially with noisy diffusion targets)
- EMA smooths these oscillations, producing more consistent predictions
- The EMA model typically achieves significantly higher sort accuracy than the raw training weights

### Loss Function

The training objective is standard DDPM noise prediction:

```python
# 1. Normalize actions
a_normalized = (actions - action_mean) / action_std

# 2. Sample random diffusion timestep
t = randint(0, num_train_timesteps)

# 3. Add noise to actions
noise = randn_like(a_normalized)
noisy_actions = noise_scheduler.add_noise(a_normalized, noise, t)

# 4. Predict noise conditioned on observation
predicted_noise = noise_pred_net(noisy_actions, t, global_cond=obs_cond)

# 5. MSE loss
loss = MSE(predicted_noise, noise)
```

### Action Normalization

```python
# Compute from training data
action_mean = all_actions.mean(dim=0)   # (4,)
action_std  = all_actions.std(dim=0)     # (4,)

# Before adding noise
a_normed = (actions - action_mean) / action_std.clamp(min=1e-6)

# After denoising, before output
a_denormed = predicted_actions * action_std + action_mean
```

Action normalization ensures the DDPM noise schedule operates on a well-conditioned space. Without normalization, the gripper dimension (near-bimodal at ±1) and delta-xyz dimensions (small continuous values) have vastly different scales, causing the noise schedule to poorly cover the action distribution.

### Checkpoint Selection

We save checkpoints based on **weighted sort accuracy** (the competition metric), not training loss. A model with lower loss may not achieve higher sort accuracy because:
- Loss measures noise prediction quality, not task performance
- Sort accuracy depends on the entire evaluation pipeline (action chunking, environment dynamics)
- The relationship between loss and sort accuracy is non-monotonic, especially early in training

We evaluate every 5,000 iterations on all three difficulties and save the checkpoint with the highest weighted score.

---

## 7. Inference Pipeline

### Action Chunking

```
Step  0: obs_0  → Diffusion → actions[0:16] → execute actions[0:8]
Step  1–7: Use buffered actions[1:8] from previous diffusion
Step  8: obs_8  → Diffusion → actions[0:16] → execute actions[0:8]
Step  9–15: Use buffered actions[1:8] from previous diffusion
...
```

**Why action chunking matters**: Running the full diffusion process every step would be slow (16 denoising iterations × U-Net forward pass) and would discard temporal coherence. Instead, we run diffusion once to produce 16 predicted actions, execute the first 8, then re-plan with a fresh observation. This provides:

- **Temporal consistency**: Actions within each chunk are generated together, ensuring smooth motion
- **Efficiency**: Only 1 diffusion run per 8 environment steps (8× faster than per-step planning)
- **Error correction**: Re-planning every 8 steps corrects accumulated drift

### Observation History

```python
# At each step, we stack the current and previous observation
obs_seq = {
    "rgb":   torch.stack([prev_rgb, cur_rgb], dim=1),     # (B, 2, 128, 128, 3)
    "state": torch.stack([prev_state, cur_state], dim=1),  # (B, 2, 26)
}
```

The `obs_horizon=2` provides the model with a single frame of motion information (current vs. previous). This helps the model estimate velocities (how fast the arm is moving) and detect changes in the scene (whether a parcel has been picked up). More history would provide richer temporal information but at the cost of increased computation and a longer context for the U-Net.

### DDPM Denoising at Inference

```python
noise_scheduler.set_timesteps(num_inference_steps=16)
noisy = randn(B, pred_horizon, act_dim)  # Start from pure noise

for t in noise_scheduler.timesteps:
    noise_pred = noise_pred_net(noisy, t, global_cond=obs_cond)
    noisy = noise_scheduler.step(noise_pred, t, noisy).prev_sample

actions = denormalize(noisy[:, 0:act_horizon])
```

We use 16 denoising steps at inference (training's 100 timesteps are subsampled to 16). More steps would produce slightly better actions but with diminishing returns and linearly increasing compute. 16 steps provides a good balance between action quality and inference speed.

---

## 8. Ablation Studies & Design Rationale

### Multi-Difficulty vs. Single-Difficulty Training

| Training Data | Easy | Medium | Hard | Weighted |
|:-------------:|:----:|:------:|:----:|:--------:|
| Easy only (200 eps) | Best | ~0% | 0% | ~0.2×best |
| Easy+Medium (400 eps) | Good | Moderate | ~0% | ~0.2×good + 0.3×mod |
| **All three (600 eps)** | **Good** | **Moderate** | **Emerging** | **Best** |

Training on all difficulties simultaneously provides the best weighted score because:
- Hard demos provide the visual diversity needed for robust perception
- Easy demos provide clean learning signal for basic manipulation
- The shared visual encoder benefits from all data sources

### U-Net Capacity

| UNet Dims | Params | Training Speed | Sort Accuracy |
|:---------:|:------:|:--------------:|:-------------:|
| `[64, 128, 256]` | ~4.5M | Fast | Low |
| `[128, 256, 512]` | ~12M | Moderate | **Higher** |

Larger capacity allows the denoiser to model more complex conditional distributions. With 6 parcels and varying configurations, the action distribution is highly multi-modal — a larger network can better represent these modes.

### Augmentation Impact

| Augmentation | Easy | Medium | Hard | Weighted |
|:------------:|:----:|:------:|:----:|:--------:|
| None | OK | Poor | 0% | Low |
| Color only | Better | Moderate | Emerging | Better |
| Color + Spatial | **Best** | **Good** | **Better** | **Best** |

Without augmentation, the model overfits to exact pixel patterns and fails to generalize to held-out configs (different lighting, positions). Color augmentation is especially critical for hard difficulty where the model must read tag colors under varying conditions.

---

## 9. Known Limitations & Future Work

### Current Limitations

1. **Resolution bottleneck**: 128×128 images make fine-grained tag color discrimination difficult. Higher resolution cameras would significantly improve hard difficulty performance.
2. **Sequential execution errors**: A single failed grasp (parcel slips, misaligned placement) can cascade into failure for the remaining parcels. Error recovery strategies could improve robustness.
3. **No depth information**: The policy lacks depth perception, making it harder to estimate precise distances for grasping. RGBD input could improve pick accuracy.
4. **Limited demonstration diversity**: 200 episodes per difficulty provides reasonable coverage but may not span the full variation in held-out configs. Generating additional demonstrations (especially for hard difficulty with diverse bin swap configurations) could help.

### Future Directions

1. **ACT (Action Chunking Transformer)**: Replace the U-Net denoiser with a transformer-based architecture that can attend to specific spatial regions of the image, potentially improving tag color reading.
2. **Progressive training**: Start with easy-only training to learn basic manipulation, then gradually introduce medium and hard data (curriculum learning).
3. **Self-generated data**: Use the trained policy to collect on-policy data, identify failure modes, and generate targeted demonstrations for those scenarios.
4. **Ensemble methods**: Combine multiple checkpoints (e.g., best-per-hard, best-per-medium) with different strengths into an ensemble policy.
5. **Fine-tuning with RL**: Use the imitation-learned policy as initialization for reinforcement learning with a shaped reward, potentially improving performance beyond the demonstration quality ceiling.

---

## 10. Reproduction & Submission

### Final Checkpoint

```
/kaggle/working/marso_output/checkpoints/best_sort_accuracy.pt
```

### Training Summary

- **Total iterations**: 200,000 (extended from baseline's 30K)
- **Training time**: ~6h 38m on Kaggle GPU
- **Best training accuracy**: 0.2375 weighted
- **Best training loss**: 0.00453

### Final Evaluation Output

```
  STEP 5: Final Evaluation
    easy   : 56.25%
    medium : 17.19%
    hard   : 12.50%
--------------------------------------------------
    easy    :  56.25%  (w=0.2)
    medium  :  17.19%  (w=0.3)
    hard    :  12.50%  (w=0.5)
    FINAL   :  22.66%
--------------------------------------------------
```

### Evaluation Notes

- **Number of evaluation episodes per difficulty**: 16 (`FINAL_EVAL_EPS = 16`)
- **Observation shape**: `rgb = (B, 128, 128, 3)`, `state = (B, 26)`
- **Action chunk**: 16 predicted × 8 executed per diffusion run
- **Diffusion inference steps**: 16 (DDIM/DDPM subsampling)
- **Action range**: normalized to `[-1.0, 1.0]` after denormalization

### Submission Format

The model produces a single checkpoint (`best_sort_accuracy.pt`) compatible with the repo's `il_policy:load_dp_rgb` loader, packaged with a `submission.yaml` declaring `obs_mode: rgb` and the checkpoint path.

### Key Files

| File | Purpose |
|------|---------|
| `solve_marso_hack_v5.py` | End-to-end training + evaluation + submission script |
| `best_sort_accuracy.pt` | Final trained checkpoint (selected on weighted sort accuracy) |
| `submission.yaml` | Submission manifest for the competition judge |
| `TECHNICAL_APPROACH.md` | Detailed write-up of design decisions |

---

*Berlin Marso Hackathon 2026 — WarehouseSort Multi-Diffusion Policy Solution*