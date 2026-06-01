# Cosmos 3 → DK-1 action fine-tuning — plan

> Status: research + scaffold complete. Implementation not started.
> Upstream code vendored at `../cosmos-framework` (NVIDIA/cosmos-framework, shallow clone, HEAD 411d25b).
> ⚠ Cosmos 3 released 2026-05-31 — everything below is from the live repo/docs/model card, not prior knowledge.

## 1. What Cosmos 3 is (Nano tier)

- **Mixture-of-Transformers (MoT), two towers:**
  - **Reasoner** — autoregressive VLM, built on **Qwen3-VL-8B-Instruct** (confirmed in
    `configs/.../sft/models/nano_model_config.py`). Long context (≤256K), chain-of-thought, grounding.
  - **Generator** — diffusion transformer for continuous outputs (video / action / audio), flow-matching loss.
- **Cosmos3-Nano ≈ 16B trainable** (model card) = both towers (~8B reasoner + diffusion generator).
  (Blog's "8B" appears to refer to one tower; HF card 16B is authoritative.)
- **Omnimodal:** text / image / video / audio / **action**.
- Model config exposes `max_action_dim=64`, `num_embodiment_domains=32` — new embodiments are first-class.
- **Action modes:** `forward_dynamics`, `inverse_dynamics`, `policy`.
- License: **OpenMDW-1.1** (NOT Apache); precision bf16; tested on Ampere/Hopper/Blackwell.
- Tech report: https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf

## 2. Why this is a good fit for DK-1

The action stack already speaks **LeRobot**, which is exactly our data format:
- `cosmos_framework/data/vfm/action/datasets/droid_lerobot_dataset.py` — `DROIDLeRobotDataset`
  loads LeRobot parquet + videos, normalizes actions, emits chunks for forward/inverse/policy modes.
- `cosmos_framework/data/vfm/action/action_spec.py` — declarative `ActionSpec` DSL. Its own docstring
  shows **bimanual** and **joint-space** examples — directly expressible for DK-1.
- `domain_utils.get_domain_id` — registers an embodiment domain id (we add `dk1`).

DK-1 action layout (from FastWAM): 14D = left arm joints[0..5] + left gripper[6] + right arm joints[7..12]
+ right gripper[13]. As an ActionSpec:
```python
build_action_spec(Joint(n=6, label="left_arm"),  Gripper(prefix="left"),
                  Joint(n=6, label="right_arm"), Gripper(prefix="right"))   # → 14D
```
(Cartesian variant also possible via Pos/Rot if we prefer EE-space, but joint-space matches the
working FastWAM head and avoids the IK/URDF issues we hit there.)

## 3. The DK-1 datamix (reuse FastWAM's)

Same 21-source blend used for `dk1-pretrain-40d-30hz-joints`:
- DK-1 teleop: black_and_white_swan, cutlery_basket, duplo (in_box / sorting / disassembly / stack)
- `dk1-merge-2026-03` (30Hz)
- RoboTwin synthetic: stack_blocks_two / three (17Hz)
- 3 cameras (head, left_wrist, right_wrist), ~30Hz, pos/vel/torque state.
All already on disk under `/workspace/data/` and HF (`andreaskoepf/*`).

## 4. Adaptation work items

1. **DK-1 embodiment** — `action_spec` (14D bimanual joint-space) + a `dk1` domain id.
2. **`DK1LeRobotDataset`** — adapt `DROIDLeRobotDataset`:
   - image features `observation.images.{head,left_wrist,right_wrist}` (DROID uses different keys),
   - 14D joint actions (DROID is 10D cartesian single-arm),
   - blend across the 21 sources (DROID loads one root),
   - `policy` mode for behavior cloning (predict action chunk from obs).
3. **Action normalization** — Cosmos uses `action_normalization.load_action_stats` + a JSON. Generate a
   DK-1 stats JSON in Cosmos format (we have FastWAM stats but the schema differs — recompute).
4. **SFT experiment config + TOML** — template from `vision_sft_nano` / the action integration example
   (`examples/integration/trainer_level_training.py` documents the training_step batch contract).
   `[job].task="vfm"`, action-gen enabled, our dataset + embodiment.
5. **Checkpoint prep** — download `nvidia/Cosmos3-Nano` (~30GB DCP) and
   `python -m cosmos_framework.scripts.convert_model_to_dcp`.
6. **Launch** — `torchrun -m cosmos_framework.scripts.train --sft-toml=<dk1.toml>`.

## 5. Hard constraints / open questions

- **Compute:** their SFT is tested on **8× H100 80GB**; we have **2× H100**. Full 16B FSDP SFT on 2 GPUs is
  not their tested regime. **LoRA is supported** (`lora_enabled=true, rank=16, alpha=32`, generator MoE
  attn targets) and is the realistic path — verify it fits + trains on 2 GPUs first.
- **Mode:** `policy` (behavior cloning) is the obvious target for a robot policy; but Cosmos's headline is
  world+action — do we also want video/world-model fine-tuning, or action-only?
- **Install:** `cosmos_framework` must be `pip install -e .` (deep internal imports; cu130, vLLM extras).
  Heavy deps; needs its own venv.
- **License:** OpenMDW-1.1 governs the weights — check redistribution terms before uploading derivatives.

## 6. Immediate next steps (proposed)

1. Set up a `cosmos_framework` venv and `pip install -e ../cosmos-framework`; smoke-test the integration
   demo (random-init) to confirm the env works on this box.
2. Download Cosmos3-Nano + convert to DCP.
3. Implement the DK-1 embodiment + `DK1LeRobotDataset` + stats.
4. Write the LoRA action-SFT TOML; dry-run on 2 GPUs at tiny scale.
