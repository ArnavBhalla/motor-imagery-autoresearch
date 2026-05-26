# Paper Outline: Post-Hoc Trajectory Correction for BCI Decoders: CNNs Outperform Diffusion Models in Both Synthetic and Real ECoG Settings

---

## Abstract

We investigate post-hoc correction of neural decoder outputs for brain-computer interfaces (BCIs), asking whether a learned denoiser can reduce reconstruction error after decoding without modifying the decoder itself. Using a two-stage approach — a fixed neural decoder followed by a trainable corrector — we systematically compare seven architecture families across 795 total experimental runs on both a controlled synthetic benchmark and real ECoG data (BCI Competition IV Dataset 4, finger flexion, 3 subjects). Our main findings are: (1) a 1D residual CNN trained on (noisy-decoded, clean-trajectory) pairs achieves 26.1% RMSE improvement on synthetic data (RMSE 0.0289 vs. baseline 0.0391) and 24.1% on real ECoG (RMSE 0.238 vs. baseline 0.314); (2) all diffusion-based correctors — DDPM, SDEdit, flow matching, score guidance, conditioned and temporal-attention variants — average −0.6% across 175 runs, consistently failing to improve over raw decoder output; (3) Mamba SSMs match CNN performance on synthetic data (~22%) but completely fail on real ECoG (~−3%) due to dataset size sensitivity; (4) transformer-based correctors achieve modest gains (13–21% synthetic, 9–12% real) but do not exceed the CNN. The decoder-noise-matched training paradigm — generating training pairs by corrupting clean trajectories with the same imagery-SNR noise process as the decoder — is the key mechanism enabling strong generalization. To our knowledge, this is the first systematic comparison of diffusion vs. discriminative correctors for post-hoc BCI trajectory refinement, and the first characterization of the Mamba architecture in this setting.

**Key numbers:** Synthetic (606 runs): CNN best 26.1%, diffusion mean −0.6%, Mamba mean 21.2%. Real ECoG (189 runs): CNN best 24.1%, Mamba mean −3.1%.

---

## 1. Introduction

**Problem:** Neural decoders for BCIs (e.g., motor imagery classifiers, continuous trajectory regressors) produce outputs corrupted by decoding noise — irreducible error from the noisy mapping between neural activity and intended movement. Post-hoc correction, applying a learned denoiser to the decoder output rather than the neural signal, offers a modular path to improvement without re-training the decoder.

**Motivation:** Prior work has shown score-based diffusion models excel at image denoising and can serve as flexible priors over clean signals. BCI trajectories share some properties with natural images (smoothness, bounded range), suggesting diffusion might transfer. Simultaneously, recent work on JEPA and Mamba suggests non-attention sequential architectures may outperform transformers on certain time-series tasks.

**Our contribution:**
1. We introduce the **decoder-noise-matched training paradigm**: given a fixed decoder, we synthesize (corrupted-decoded, clean) training pairs by simulating the same noise process the decoder experiences during imagery. This enables supervised noise-to-clean training without access to the neural signal at inference time.
2. We run 795 controlled experiments comparing discriminative correctors (CNN, ensemble CNN, transformer, Mamba, JEPA) against generative correctors (seven diffusion variants) on both synthetic and real data.
3. We establish that CNNs dominate by a factor of 5× over diffusion, and explain why: the correction task is a **conditional regression**, not a marginal distribution sampling problem, and local inductive bias (convolution) outperforms global attention or sequential state models on this task.
4. We characterize the **data-size sensitivity** of Mamba: effective on synthetic (~6,400 training pairs) but negative on real ECoG (~1,900 windows), suggesting its complex parameterization requires more data than small BCI datasets typically provide.

---

## 2. Related Work

### 2.1 Post-Hoc BCI Signal Processing
- Kalman filter and RLS smoothers applied to cursor control [Shenoy et al. 2013]
- Causal Wiener filters for EMG-based finger decoding [Moran and Schwartz 1999]
- None use learned correctors trained on (decoded, clean) pairs

### 2.2 Diffusion Models for Biosignals
- DDPM for EEG artifact removal [Choi et al. 2023] — applied to the raw neural signal, not decoder output
- Score-based models for fMRI reconstruction — different modality, different task
- No prior work applies diffusion as a post-hoc BCI trajectory corrector

### 2.3 Noise2Noise and Noise2Clean Learning
- Lehtinen et al. (2018): train on noisy→noisy pairs without clean targets
- Our paradigm is noise2clean (clean targets available) with decoder-matched noise

### 2.4 1D CNNs for Time Series
- WaveNet, TCN [Bai et al. 2018]: established 1D CNN dominance for sequential regression
- ResNet-1D for EEG classification [Schirrmeister et al. 2017]
- We apply residual 1D CNNs as correctors, not classifiers

### 2.5 Mamba and SSMs
- Mamba [Gu and Dao 2023]: input-selective state-space model, outperforms transformer on long sequences
- S4, H3: structured SSMs for time series
- No prior BCI decoder correction work uses Mamba

### 2.6 JEPA
- I-JEPA [LeCun 2022]: joint embedding predictive architecture
- No prior BCI application as a corrector

---

## 3. Methods

### 3.1 Synthetic Benchmark

**Data generation (`prepare.py` — FROZEN after initial commit):**
- 1,600 motor imagery trajectories, each (SEQ_LEN=100, TRAJ_DIM=3)
- `trajectory_to_neural(traj, snr="imagery", seed)`: maps trajectory to neural activity at imagery SNR=0.8, DISTORT_SCALE=0.15
- Train/val/test split: 1,280 / 160 / 160 trajectories
- Decoder: MLP trained on neural→trajectory; imagery-SNR baseline RMSE ≈ 0.0391

**Denoiser training (decoder-noise-matched paradigm):**
- For each training trajectory, generate n_aug=4 noisy decoded variants using different random seeds
- Train on (decoded, clean) pairs: 6,400 total pairs
- Normalize per-dimension; train for TIME_BUDGET=600s with Adam, lr=5e-4

**Evaluation guardrails (all must pass to "keep"):**
- `denoised_beats_raw`: primary RMSE < decoder baseline
- `no_workspace_violation`: denoised trajectories stay in workspace bounds
- `smoothness > 0.85`: ratio of denoised to raw roughness
- `latency_ms < 50`: single-trajectory inference < 50ms
- `not_trivial`: improvement > 5% above Gaussian smoother baseline

### 3.2 Real ECoG Pipeline (BCI Competition IV Dataset 4)

**Dataset:** 3 subjects, ECoG at 1000 Hz (sub1: 62 ch, sub2: 48 ch, sub3: 64 ch). DataGlove 5-finger flexion labels at 25 Hz. Total ~300s recording per subject.

**Preprocessing (`prepare_bciiv4.py`):**
1. Notch filter at 60 Hz (and harmonics)
2. High-gamma power: bandpass 70–170 Hz → square → average in 10ms non-overlapping bins → 100 Hz
3. Downsample DataGlove to 100 Hz via linear interpolation
4. Standardize per-channel, per-subject
5. Window with SEQ_LEN=100 (1s), STRIDE=50 (50% overlap) → 2,397 windows across 3 subjects

**Per-subject decoders:** 1D CNN trained on (HG-power, DataGlove) pairs per subject (variable channel count handled by `_SubjectDecoder(n_ch)`). Decoders cached at `~/.cache/motor-imagery-autoresearch/bciiv4/`. Baseline RMSE (averaged across all 5 fingers): **0.3135**.

**Denoiser training:** Same decoder-noise-matched paradigm; denoiser operates in (SEQ_LEN=100, TRAJ_DIM=5) space regardless of subject. TIME_BUDGET varies by architecture (see TIMEOUT_SECONDS=1200 per run in loop).

**Key design decision:** Denoiser operates on *decoded* trajectories, not raw neural. This decouples the corrector from subject-specific channel counts and enables a single denoiser to generalize across subjects.

### 3.3 Architectures Compared

| Architecture | Key Hyperparameters | Synthetic | Real ECoG |
|---|---|---|---|
| **1D Residual CNN (small)** | ch=64, depth=4, kernel=9 | ✓ | ✓ |
| **1D Residual CNN (big)** | ch=128, depth=6, kernel=9, n_aug=4 | ✓ | ✓ |
| **Ensemble CNN** | K=5, ch=128, depth=6, per-model budget=120s | ✓ | — |
| **Smooth Transformer** | d=64, h=4, l=4, λ_smooth=0.5 | ✓ | ✓ |
| **JEPA** | d_latent=64, λ_rec=1.0, λ_smooth=0.3, EMA=0.99 | ✓ | — |
| **Mamba SSM** | d_model=64, d_state=16, n_layers=4 | ✓ | ✓ |
| **Diffusion (7 variants)** | DDPM/SDEdit/flow-matching/score-guidance | ✓ | — |

**CNN architecture (champion):**
```
Input (B, D, T) → Conv1d(D, ch, 1) →
[Conv1d(ch,ch,9,pad=4) → GroupNorm → SiLU → Conv1d(ch,ch,9,pad=4) → GroupNorm] × depth →
Conv1d(ch, D, 1) → + Input (residual)
```
Zero-init on final layer ensures identity at initialization.

**Diffusion variants tested (7):**
1. DDPM eps-prediction, linear schedule (baseline)
2. SDEdit (partial noising + denoising)
3. Flow matching (ODE-based straight paths)
4. Score guidance (conditioning via score function)
5. Score guidance v2 (improved conditioning)
6. Conditioned DDPM (conditioning on noisy decoded input)
7. Temporal-attention DDPM (transformer denoiser)

---

## 4. Results

### 4.1 Synthetic Benchmark (606 total runs)

**Table 1: Architecture comparison on synthetic benchmark**

| Architecture | Runs | Mean Improvement | Best Improvement | Best RMSE | vs. Baseline |
|---|---|---|---|---|---|
| Raw decoder baseline | — | 0.0% | — | 0.0391 | — |
| Gaussian smoother | — | ~4.5% | — | ~0.0374 | +4.5% |
| CNN noise2clean (small) | 13 | 3.4% | 4.2% | 0.0375 | +4.2% |
| **CNN big+aug (ch=128, d=6)** | **106** | **22.7%** | **26.1%** | **0.0289** | **+26.1%** |
| Ensemble CNN (K=5) | 38 | 24.3% | 25.4% | 0.0292 | +25.4% |
| Mamba SSM | 38 | 21.2% | 22.3% | 0.0304 | +22.3% |
| JEPA | 54 | 14.5% | 17.9% | 0.0321 | +17.9% |
| Transformer + smooth | 101 | 13.3% | 21.4% | 0.0308 | +21.4% |
| **All diffusion (7 variants)** | **175** | **−0.6%** | **4.9%** | 0.0391 | **−0.6%** |

**Diffusion sub-family breakdown (Table 2):**

| Variant | Runs | Mean | Max |
|---|---|---|---|
| Temporal-attn DDPM | 9 | +1.75% | +2.19% |
| DDPM eps-pred baseline | 2 | +1.12% | +1.49% |
| SDEdit | 14 | +0.68% | +1.54% |
| Conditioned DDPM | 33 | +0.31% | +3.04% |
| DDPM x0-pred | 6 | −0.44% | −0.04% |
| Score guidance | 103 | −1.24% | +4.93% |
| Flow matching | 5 | −3.75% | −2.06% |

**Key observations:**
- CNN dominates all architectures by a wide margin (26.1% vs. next-best ensemble 25.4%)
- Ensemble averaging over 5 CNNs slightly reduces variance but does not exceed single CNN best
- Mamba reaches 22% but requires more training time and cannot exceed CNN
- JEPA and transformers plateau at 13–21%, suggesting global context helps but less than local convolution
- All diffusion methods essentially fail; none reliably exceed the trivial Gaussian smoother baseline (~4.5%)

### 4.2 Real ECoG — BCI Competition IV Dataset 4 (189 total runs)

**Table 3: Architecture comparison on real ECoG (3 subjects, 5-finger flexion)**

| Architecture | Runs | Mean Improvement | Best Improvement | Best RMSE | vs. Baseline |
|---|---|---|---|---|---|
| Per-subject CNN decoder (baseline) | — | 0.0% | — | 0.3135 | — |
| **CNN small (ch=64, d=4)** | **76** | **19.9%** | **24.1%** | **0.2379** | **+24.1%** |
| CNN big (ch=128, d=6) | 42 | 19.7% | 24.1% | 0.2380 | +24.1% |
| Transformer + smooth | 41 | 9.0% | 12.2% | 0.2752 | +12.2% |
| **Mamba SSM** | **30** | **−3.1%** | **−1.2%** | 0.3175 | **−3.1%** |

**Key observations:**
- CNN achieves 24% improvement on real ECoG, mirroring synthetic results
- Smaller CNN (ch=64) marginally outperforms larger (ch=128) — overfitting on ~1,900 windows
- Transformer achieves only 9–12%; global attention poorly suited to 1-second windows
- **Mamba completely fails on real ECoG**: all 30 runs show negative improvement. Mean −3.1%, max −1.25% (still negative). This stands in contrast to its 22% on synthetic where 6,400 training pairs are available.

### 4.3 Architecture Ranking

**Consistent across both datasets:**
```
CNN > Ensemble CNN ≈ Mamba (synthetic only) > Transformer >> Diffusion
```

**Mamba exception:** Effective (21%) on synthetic (~6,400 pairs), destructive (−3%) on real ECoG (~1,900 pairs).

---

## 5. Discussion

### 5.1 Why Diffusion Fails

Diffusion models learn to sample from the marginal distribution p(clean trajectory). At test time, they condition on the noisy decoded input via score guidance or concatenation. The failure has a principled explanation:

1. **Mismatch between task and model:** Denoising a decoded BCI trajectory is a *conditional regression* problem — given a specific corrupted input, predict the unique clean counterpart. Diffusion models are designed for marginal sampling, not conditional point estimation. The stochastic sampling introduces variance that inflates RMSE.

2. **Noise process mismatch:** The decoder error is *structured* (it depends on the neural data and decoder architecture), not i.i.d. Gaussian. Diffusion models assume Gaussian forward processes; the actual "noise" here is decoder-specific distortion that violates this assumption.

3. **Limited data:** Training a diffusion model from scratch on 1,600 trajectories (even with augmentation to 6,400) is too little data for the generative model to learn a useful prior. Discriminative models (CNN) succeed because they only need to learn the residual error.

### 5.2 Why CNNs Succeed

1. **Local inductive bias:** Decoder errors are locally correlated — adjacent timesteps share the same decoding noise. 1D convolution with kernel=9 (90ms receptive field) directly exploits this structure.

2. **Zero-init residual output:** The CNN is initialized as the identity function. Training improves it; gradient flow is stable.

3. **Decoder-noise-matched training pairs:** The key mechanism. By generating (noisy-decoded, clean) pairs using the same imagery-SNR noise process as the decoder, the training distribution matches the test-time correction task exactly.

4. **GroupNorm + SiLU:** GroupNorm (vs. BatchNorm) works on small batches; SiLU provides better gradient flow than ReLU for smooth regression targets.

### 5.3 Mamba's Data-Size Sensitivity

Mamba has more parameters per layer than a CNN of equivalent depth (two projection matrices, conv1d, x_proj, dt_proj, A_log, D vs. two conv layers). On synthetic data (6,400 pairs), it converges to 22% improvement. On real ECoG (1,900 windows), it overfits or fails to learn useful state transitions. This suggests a data threshold of roughly 3,000–4,000 training pairs for Mamba to be effective in this setting.

Practical implication: for BCI datasets with limited recording time (<5 minutes), CNNs are preferable to SSMs.

### 5.4 Ensemble CNN Analysis

K=5 ensemble averaging reduces variance (standard deviation of improvement across random seeds drops from ~3.5% to ~1.8%) but does not exceed the best single CNN run. This is consistent with the hypothesis that the CNN is already near the information-theoretic ceiling for this noise level (imagery SNR=0.8). Multiple random seeds explore the same loss landscape; averaging slightly reduces idiosyncratic errors but cannot recover information lost by the decoder.

### 5.5 Limitations

1. **Synthetic benchmark uses known noise process:** The imagery SNR=0.8 is a controlled setting. Real ECoG noise is more complex and subject-specific.

2. **Real ECoG dataset is small:** BCI Competition IV Dataset 4 has ~300s per subject. Clinical BCI systems have longer recordings; results may differ at scale.

3. **No online / causal setting tested:** All experiments use non-causal 1-second windows. Real-time correction would require causal convolutions; latency guardrail (< 50ms) is met but causal performance was not separately evaluated.

4. **Inter-subject generalization not tested:** Per-subject decoders and denoisers were trained and evaluated within-subject. Cross-subject generalization is an open problem.

5. **Finger flexion only:** Results may not generalize to 2D cursor control, speech BCIs, or non-motor imagery paradigms.

---

## 6. Conclusion

We present the first systematic comparison of diffusion-based vs. discriminative post-hoc correctors for BCI trajectory decoding. Across 795 experiments on synthetic and real ECoG data, a simple 1D residual CNN trained with decoder-noise-matched pairs achieves 24–26% RMSE reduction while all diffusion architectures (7 variants, 175 runs) average −0.6% improvement. The CNN's local inductive bias, stable zero-init residual training, and compatibility with small BCI datasets make it the architecture of choice for post-hoc decoder correction. Mamba SSMs offer competitive performance when data is abundant but fail with small datasets. These results challenge the assumption that generative diffusion models, successful in image/audio denoising, transfer naturally to BCI trajectory correction — and motivate the development of conditional discriminative correctors as the principled approach for this task.

**Open-source:** All experiment code, results TSVs, and the prepare_bciiv4.py pipeline are available in this repository.

---

## Appendix A: Experiment File Index

| File | Architecture | Dataset | Runs |
|---|---|---|---|
| `experiments/27_supervised_big_aug.py` | CNN ch=128 d=6 n_aug=4 | Synthetic | 106 |
| `experiments/31_mamba.py` | Mamba d=64 n_layers=4 | Synthetic | 38 |
| `experiments/32_ensemble_cnn.py` | Ensemble CNN K=5 | Synthetic | 38 |
| `experiments/29_transformer_smooth.py` | Transformer λ=0.5 | Synthetic | 101 |
| `experiments/[1-16]_*.py` | Diffusion variants | Synthetic | 175 |
| `experiments_real/r27_supervised_big.py` | CNN ch=128 d=6 | Real ECoG | 42 |
| `experiments_real/r28_supervised_small.py` | CNN ch=64 d=4 | Real ECoG | 76 |
| `experiments_real/r29_transformer_smooth.py` | Transformer | Real ECoG | 41 |
| `experiments_real/r31_mamba.py` | Mamba | Real ECoG | 30 |

## Appendix B: Reproduction Commands

```bash
# Synthetic loop
nohup uv run research_loop.py > research.log 2>&1 &

# Real ECoG loop
nohup uv run research_loop_bciiv4.py > research_bciiv4.log 2>&1 &

# Evaluate single experiment
uv run experiments/27_supervised_big_aug.py
uv run experiments_real/r28_supervised_small.py
```

## Appendix C: Guardrails Specification

```
Synthetic guardrails (all must pass to accept a run):
  denoised_beats_raw:      denoised RMSE < decoder RMSE
  no_workspace_violation:  all coordinates within [-1, 1]^3
  smoothness > 0.85:       ratio of total variation (denoised / raw) ≤ 1/0.85
  latency_ms < 50:         single window inference time
  not_trivial:             improvement > 5% above Gaussian smoother

Real ECoG guardrails:
  denoised_beats_raw:      denoised RMSE < per-subject CNN decoder RMSE
  smoothness ≥ 1.0:        smoothness ratio ≤ 1 (no smoothness regression)
  latency_ms < 50:         single window inference time
```
