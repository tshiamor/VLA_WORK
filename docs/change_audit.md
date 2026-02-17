# Change Audit: VLA Pipeline Modifications

Audit of all changes made during the diagnosis and Isaac Lab integration sessions, relative to the initial commit (`f953640`).

---

## 1. Summary

**9 files modified** (tracked by git), **13 new files added** (untracked).

All changes are on the **VLM + Action Expert stream only** — the single Qwen2.5-VL + Flow Matching architecture. There are **no OpenVLA, Pi Zero, or Groot** implementations in this codebase. The architecture is monolithic: one VLA model, one training pipeline.

---

## 2. VLA Model Core — What Changed

### `vla/models/vla_model.py` (+164 / -36 lines)

| Change | Why | Impact |
|--------|-----|--------|
| `forward()` gained `context_images`, `wrist_images`, `instructions` params | Enable batched training (trainer passes tensors, not PIL) | **Additive** — old `images`/`instruction`/`vlm_inputs` paths untouched |
| `predict_action()` casts condition to `flow_dtype` before sampling | Fix dtype mismatch (bfloat16 VLM output → flow head) | **Bug fix** — was crashing at inference |
| `predict_action()` denormalizes in float32 | Precision fix for action denormalization | **Bug fix** — minor |
| `load_checkpoint()` now supports 3 formats: `projector_state_dict` (trainer lightweight), `projector` (model lightweight), `model_state_dict` (legacy full) | Support checkpoints from both trainer and model save paths | **Backward compatible** — old format still works |
| `torch.load()` calls use `weights_only=False` | PyTorch 2.7 defaults to `weights_only=True`, fails on numpy arrays in checkpoints | **Bug fix** for PyTorch 2.7 |
| `save_checkpoint()` checks `hasattr(self.vision_encoder, 'lora_modules')` | Prevent crash when LoRA not enabled | **Safety fix** |
| New `load_vla_for_inference()` helper function | Convenience function for loading checkpoint → eval-ready model | **Additive** — new function, nothing depends on it yet |
| `create_vla_model()` gained `proprio_dim` parameter (default=27) | Pass proprio_dim through to ActionExpertConfig | **Backward compatible** — default matches original hardcoded value |

### `vla/models/vision_encoder.py` (+228 / -26 lines)

| Change | Why | Impact |
|--------|-----|--------|
| **LoRA forward hooks added** (`register_forward_hook`) | **Critical bug fix** — LoRA layers were trained but never applied during forward pass | **Bug fix** — this was the main discovery |
| `forward()` gained `context_images`, `wrist_images`, `instructions` params | Enable batched training path | **Additive** — old `images`/`instruction`/`inputs` paths untouched |
| New `_forward_batch()` method | Processes batch by iterating samples through VLM individually | **Additive** — only called from new batched path |
| `_extract_outputs()` refactored | Returns `last_hidden_state[:, -1:, :]` (last token only) + pooled output, consistent between single and batch paths | **Behavior change** — was returning full sequence, now returns last token |
| `prepare_inputs()` handles numpy/tensor images | Auto-convert numpy arrays and tensors to PIL | **Additive** — PIL images still work |
| `_tensor_to_pil()` helper | Used by batched path and prepare_inputs | **Additive** |
| Optional 4-bit quantization support | `use_4bit` config option with BitsAndBytes | **Additive** — off by default |
| Image labels default to `["Context view", "Wrist view"]` | Better semantic labels | **Cosmetic** |

### `vla/models/flow_matching.py` (+12 / -5 lines)

| Change | Why | Impact |
|--------|-----|--------|
| `SinusoidalPosEmb` preserves input dtype | Was creating float32 embeddings regardless of input dtype | **Bug fix** — enables bfloat16 sampling |
| `sample()` creates noise/timesteps in `condition.dtype` | Was using default float32, causing dtype mismatch with bfloat16 condition | **Bug fix** |

### `vla/models/action_expert.py` — **UNCHANGED**

### `vla/models/projector.py` — **UNCHANGED**

---

## 3. Training Pipeline — What Changed

### `vla/training/trainer.py` (+163 / -21 lines)

| Change | Why | Impact |
|--------|-----|--------|
| `autocast('cuda', ...)` instead of `autocast(...)` | PyTorch 2.7 deprecation of positional device arg | **Compatibility fix** |
| Trainer calls `model(context_images=..., wrist_images=..., instructions=...)` | Use batched path instead of single-sample | **Interface change** — matches new `forward()` signature |
| `_save_checkpoint()` saves lightweight by default | ~100MB instead of ~16GB per checkpoint | **Behavior change** — saves only trainable weights |
| `_save_checkpoint()` saves dataset normalization stats | `proprio_mean/std`, `dataset_action_mean/std` | **Additive** |
| `load_checkpoint()` supports both lightweight and full formats | Can resume from either checkpoint type | **Backward compatible** |

### `scripts/train.py` (+110 / -12 lines)

| Change | Why | Impact |
|--------|-----|--------|
| Added `--hdf5_path` argument | Support HDF5 datasets alongside folder-based | **Additive** — `--data_dir` still works |
| Added `--proprio_dim`, `--key_mapping_file`, `--instruction` args | Configuration for HDF5 pipeline | **Additive** |
| `create_hdf5_dataloader` path | New dataloader for HDF5 format | **Additive** — folder path untouched |
| Stores dataset normalization stats on model | Allows inference scripts to denormalize actions | **Additive** |

---

## 4. Data Pipeline — What Changed

### `vla/data/__init__.py` (+7 / -2 lines)

| Change | Why | Impact |
|--------|-----|--------|
| Exports `HDF5VLADataset`, `create_hdf5_dataloader`, `verify_hdf5_dataset`, `create_dataloader` | Make new data classes importable | **Additive** |

### `vla/data/hdf5_dataset.py` — **NEW FILE** (untracked)

New HDF5 dataset implementation for NVIDIA Cosmos-format data. Does not modify the existing `vla/data/dataset.py`.

---

## 5. Init/Export Changes

### `vla/__init__.py` (+2 lines)
- Exports `load_vla_for_inference`

### `vla/models/__init__.py` (+2 lines)
- Exports `load_vla_for_inference`

---

## 6. New Files (Untracked)

| File | Purpose | Imports VLA core? |
|------|---------|-------------------|
| `configs/key_mappings/nvidia_cosmos.json` | Key mapping for NVIDIA Cosmos dataset | No |
| `configs/training/cube_stacking_cosmos.yaml` | Training config for cube stacking | No |
| `isaaclab_ext/scripts/run_vla_cube_stacking.py` | Isaac Lab closed-loop inference with GUI | Yes |
| `scripts/diagnose_vla.py` | VLA diagnostic tests | Yes |
| `scripts/diagnose_conditioning.py` | VLM conditioning diagnostics (NEW) | Yes |
| `scripts/overfit_test.py` | Overfit test on 5 demos | Yes |
| `scripts/run_cube_stacking.py` | Offline inference on dataset | Yes |
| `scripts/verify_dataset.py` | HDF5 dataset verification | Yes |
| `scripts/inspect_nvidia_dataset.py` | Raw HDF5 structure inspection | No (h5py only) |
| `scripts/setup_env.sh` | Conda env setup script | No |
| `scripts/train_cube_stacking.sh` | Shell wrapper for training | No |
| `scripts/train_mcx_card.sh` | Shell wrapper for MCX card training | No |
| `vla/data/hdf5_dataset.py` | HDF5 dataset class | Yes |
| `docs/diagnosis_overfit_test.md` | Diagnosis report | No |

---

## 7. Impact on Existing Scripts

### Scripts that are **SAFE** (unaffected):

| Script | Status | Notes |
|--------|--------|-------|
| `scripts/collect_demos.py` | Safe | Only uses `vla.data.dataset.VLADemoDataset` — unchanged |
| `scripts/evaluate.py` | **Needs fix** | Uses `torch.load()` without `weights_only=False` — will fail on PyTorch 2.7 with numpy-containing checkpoints |
| `scripts/visualize.py` | **Needs fix** | Same `torch.load()` issue as evaluate.py |
| `scripts/run_cube_stacking.py` | Safe | Uses new batched `model()` API (`context_images`/`wrist_images`/`instructions`) — already compatible |
| `isaaclab_ext/scripts/run_vla_lift.py` | Safe | Conditional VLA import, uses `predict_action()` which is unchanged |

### Scripts with known issues:

1. **`scripts/evaluate.py:230`** — `torch.load(checkpoint_path, map_location='cpu')` needs `weights_only=False`
2. **`scripts/visualize.py:62`** — `torch.load(checkpoint_path, map_location='cpu')` needs `weights_only=False`

Both scripts also use `create_vla_model()` without `proprio_dim` — this is fine because `proprio_dim=27` is the default.

Both scripts use `model.predict_action(images=..., instruction=..., ...)` — this API is **unchanged** and works correctly.

---

## 8. Behavioral Changes Summary

### What is different in the VLA model now vs. initial commit:

1. **LoRA actually works during forward pass** — previously trained but never applied (critical bug fix)
2. **`_extract_outputs()` returns last token** instead of full sequence for `last_hidden_state` — this changes what the projector sees
3. **Flow matching sampling works in bfloat16** — previously crashed or produced wrong results due to dtype mismatch
4. **Checkpoints are lightweight by default** — ~100MB instead of ~16GB

### What is identical:

- `ActionExpert` architecture (6-layer transformer, 16 queries)
- `VLMProjector` architecture (2-layer MLP, 3584→512)
- `FlowMatchingActionHead` architecture (OT-CFM, Euler solver, 100 steps)
- `predict_action()` API signature
- `prepare_inputs()` conversation format for Qwen VLM
- Folder-based `VLADataset` / `VLADemoDataset`
- All config dataclasses (VisionEncoderConfig, ActionExpertConfig, FlowMatchingConfig, etc.)

---

## 9. Are OpenVLA / Pi Zero / Groot Affected?

**No.** There are no OpenVLA, Pi Zero, Pi0, or Groot implementations in this codebase. The entire `VLA_WORK` repository contains a single VLA architecture:

```
Qwen2.5-VL-7B-Instruct (frozen + LoRA)
  → VLMProjector (3584 → 512)
  → ActionExpert (6-layer transformer)
  → FlowMatchingActionHead (OT-CFM)
  → Action chunks [16 × 7]
```

All changes are scoped to this architecture only.
