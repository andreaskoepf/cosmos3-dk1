# SPDX-License-Identifier: Apache-2.0
# Vendored from github.com/open-thought/fastwam (model/cartesian_fk.py), Apache-2.0.
"""Forward kinematics for the DK-1 dual-arm follower.

Wraps pytorch_kinematics over the dk1-dual-arm-urdf repo. Builds two serial
chains (left + right arm, 6 revolute joints each, ending at tool0). The dataset
14-D action / state vector is mapped to per-arm 6-D joint inputs via name match
against `action.names` from info.json.

Action / state layout we expect (verified against dk1_black_and_white_swan):
    [left_joint_1..6, left_gripper, right_joint_1..6, right_gripper]  (14)

The URDF uses `{left,right}_joint{1..6}` (no underscore before the number). The
fixed name mapping is:
    dataset name              URDF joint
    -----------------------   ------------
    left_joint_<i>.pos        left_joint<i>
    right_joint_<i>.pos       right_joint<i>
    left_gripper.pos          (passthrough — gripper handled outside FK)
    right_gripper.pos         (passthrough)
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
import pytorch_kinematics as pk


# Fixed indices into the 14-D action / state vector (positions block).
LEFT_ARM_IDX = [0, 1, 2, 3, 4, 5]
LEFT_GRIPPER_IDX = 6
RIGHT_ARM_IDX = [7, 8, 9, 10, 11, 12]
RIGHT_GRIPPER_IDX = 13
GRIPPER_IDX = (LEFT_GRIPPER_IDX, RIGHT_GRIPPER_IDX)


def _dataset_to_urdf_joint(name: str) -> Optional[str]:
    """Map a dataset action.names entry to the URDF joint name, or None for gripper/unknown."""
    if name.endswith(".pos"):
        name = name[:-4]
    if "gripper" in name:
        return None
    m = re.fullmatch(r"(left|right)_joint_(\d+)", name)
    if m:
        return f"{m.group(1)}_joint{m.group(2)}"
    return None


def verify_action_layout(action_names: Sequence[str]) -> None:
    """Assert the 14-D action layout matches our hardcoded indexing.

    Raises ValueError on mismatch so caches can't silently encode wrong joints.
    """
    if len(action_names) != 14:
        raise ValueError(f"action_names length {len(action_names)} != 14")
    # Some sources (e.g. robotwin sim) drop the ".pos" suffix but keep the
    # identical joint ORDER, which is all the fixed-index FK relies on. Compare
    # suffix-normalized so both conventions pass.
    def _norm(n: str) -> str:
        return n[:-4] if n.endswith(".pos") else n
    expected = [
        "left_joint_1", "left_joint_2", "left_joint_3",
        "left_joint_4", "left_joint_5", "left_joint_6",
        "left_gripper",
        "right_joint_1", "right_joint_2", "right_joint_3",
        "right_joint_4", "right_joint_5", "right_joint_6",
        "right_gripper",
    ]
    if [_norm(n) for n in action_names] != expected:
        raise ValueError(
            f"action.names doesn't match expected dual-arm layout.\n"
            f"  expected (suffix-normalized): {expected}\n"
            f"  got:      {list(action_names)}"
        )


@dataclass
class FKResult:
    """Per-arm forward-kinematics output, all batched on dim 0.

    Convention: quaternions in (w, x, y, z) (pytorch_kinematics default).
    """
    left_pos: torch.Tensor      # [..., 3]
    left_quat: torch.Tensor     # [..., 4] wxyz
    right_pos: torch.Tensor     # [..., 3]
    right_quat: torch.Tensor    # [..., 4] wxyz


@dataclass
class IKResult:
    """Per-arm inverse-kinematics output + diagnostics."""
    joints_14: torch.Tensor              # [B, 14] solved joints; grippers passthrough from seed
    left_converged: torch.Tensor         # [B] bool
    right_converged: torch.Tensor        # [B] bool
    left_pos_err: torch.Tensor           # [B] meters
    right_pos_err: torch.Tensor          # [B] meters
    left_rot_err: torch.Tensor           # [B] radians
    right_rot_err: torch.Tensor          # [B] radians
    left_iterations: int                 # int (chain.solve returns scalar)
    right_iterations: int
    # Joint-jump diagnostics
    max_delta_from_seed: torch.Tensor    # [B] max |joint_target - seed_joint| over the 12 arm dims
    max_delta_in_chunk: torch.Tensor     # scalar (only meaningful when B>=2): max |joints[t+1]-joints[t]|


def _quat_wxyz_to_rot(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert (w, x, y, z) quaternion to a 3x3 rotation matrix. Batched on dim 0."""
    w, x, y, z = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]
    n = (w * w + x * x + y * y + z * z).clamp(min=1e-12).sqrt()
    w, x, y, z = w / n, x / n, y / n, z / n
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*quat_wxyz.shape[:-1], 3, 3)
    return R


def _build_transform_4x4(pos: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Build a (B, 4, 4) homogeneous transform from position + wxyz quaternion."""
    B = pos.shape[0]
    R = _quat_wxyz_to_rot(quat_wxyz)
    T = torch.zeros(B, 4, 4, device=pos.device, dtype=pos.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = pos
    T[:, 3, 3] = 1.0
    return T


class DK1ArmFK:
    """Forward kinematics for both DK-1 arms. Vectorized on GPU/CPU."""

    def __init__(self, urdf_path: str | Path,
                 left_ee: str = "left_tool0", right_ee: str = "right_tool0",
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        urdf_path = Path(urdf_path)
        with urdf_path.open("rb") as f:
            urdf_bytes = f.read()
        self.urdf_path = urdf_path
        self.urdf_sha256 = hashlib.sha256(urdf_bytes).hexdigest()
        self.left_ee_link = left_ee
        self.right_ee_link = right_ee
        self._left = pk.build_serial_chain_from_urdf(urdf_bytes, end_link_name=left_ee)
        self._right = pk.build_serial_chain_from_urdf(urdf_bytes, end_link_name=right_ee)
        self.to(device=device, dtype=dtype)
        # Pre-verify the URDF exposes 6 joints per arm in the expected order.
        lj = self._left.get_joint_parameter_names()
        rj = self._right.get_joint_parameter_names()
        expected_l = [f"left_joint{i}" for i in (1, 2, 3, 4, 5, 6)]
        expected_r = [f"right_joint{i}" for i in (1, 2, 3, 4, 5, 6)]
        if lj != expected_l or rj != expected_r:
            raise ValueError(
                f"URDF joint order unexpected.\n  left:  {lj}\n  right: {rj}"
            )
        # IK solvers — lazy init on first solve_ik() call.
        self._left_ik = None
        self._right_ik = None

    def to(self, *, device, dtype):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.dtype = dtype
        self._left = self._left.to(device=self.device, dtype=self.dtype)
        self._right = self._right.to(device=self.device, dtype=self.dtype)
        return self

    @torch.no_grad()
    def forward(self, joints_14: torch.Tensor) -> FKResult:
        """Compute EE pose for both arms (no-grad — for cache generation).

        Args:
            joints_14: [B, 14] joint angles in the dataset's action layout
                (positions block). Grippers in cols 6 and 13 are ignored —
                they don't affect tool0 FK.

        Returns:
            FKResult with [B, ...] tensors.
        """
        return self._forward_impl(joints_14)

    def forward_diff(self, joints_14: torch.Tensor) -> FKResult:
        """Same FK computation as `forward`, but autograd-active.

        Use in training loops where gradients must flow from EE pose back
        through joint angles (e.g. cartesian auxiliary loss). The chain's
        forward_kinematics is fully differentiable; only the public
        `forward` wrapper above suppresses gradients for cache generation.
        """
        return self._forward_impl(joints_14)

    def _forward_impl(self, joints_14: torch.Tensor) -> FKResult:
        if joints_14.dim() != 2 or joints_14.shape[1] != 14:
            raise ValueError(f"joints_14 must have shape [B, 14], got {tuple(joints_14.shape)}")
        q = joints_14.to(device=self.device, dtype=self.dtype)
        q_left = q[:, LEFT_ARM_IDX]
        q_right = q[:, RIGHT_ARM_IDX]
        left_mat = self._left.forward_kinematics(q_left).get_matrix()    # [B, 4, 4]
        right_mat = self._right.forward_kinematics(q_right).get_matrix()
        from pytorch_kinematics.transforms import matrix_to_quaternion
        return FKResult(
            left_pos=left_mat[..., :3, 3].contiguous(),
            left_quat=matrix_to_quaternion(left_mat[..., :3, :3]).contiguous(),
            right_pos=right_mat[..., :3, 3].contiguous(),
            right_quat=matrix_to_quaternion(right_mat[..., :3, :3]).contiguous(),
        )

    # ---------------------------- IK ---------------------------- #

    def _ensure_ik(self, *, max_iterations: int = 50,
                   pos_tolerance: float = 5e-4, rot_tolerance: float = 5e-3) -> None:
        """Lazy init of per-arm PseudoInverseIK solvers."""
        if self._left_ik is not None:
            return
        # Joint limits straight from the URDF (chain.low/.high).
        L_limits = torch.stack([self._left.low,  self._left.high],  dim=-1)
        R_limits = torch.stack([self._right.low, self._right.high], dim=-1)
        self._left_ik = pk.PseudoInverseIK(
            self._left,
            pos_tolerance=pos_tolerance,
            rot_tolerance=rot_tolerance,
            max_iterations=max_iterations,
            num_retries=1,
            joint_limits=L_limits,
            enforce_joint_limits=True,
        )
        self._right_ik = pk.PseudoInverseIK(
            self._right,
            pos_tolerance=pos_tolerance,
            rot_tolerance=rot_tolerance,
            max_iterations=max_iterations,
            num_retries=1,
            joint_limits=R_limits,
            enforce_joint_limits=True,
        )

    @torch.no_grad()
    def solve_ik(self, abs_cart_14: torch.Tensor,
                 seed_joints_14: torch.Tensor,
                 fallback_seed_joints_14: torch.Tensor | None = None) -> IKResult:
        """Solve IK on both arms; seed warm-start with the supplied joint state.

        Args:
            abs_cart_14: [B, 14] absolute target poses in the layout
                `[Lpos(3), Lquat(4), Rpos(3), Rquat(4)]` (quaternions wxyz).
            seed_joints_14: [14] OR [B, 14] current joint state. If [14], all B
                problems use the same seed (per-problem seeding requires
                sequential calls — see warm_solve_chunk_ik below).
            fallback_seed_joints_14: [14] joint state used as the per-arm hold
                pose when IK doesn't converge for that arm. Default = the
                warm-start seed, which inside `warm_solve_chunk_ik` is the
                previous step's solved joints — so a failing arm holds at its
                last committed value (equivalent to last-converged, since
                failures just copy the previous step). Pass a different vector
                here only if you want an explicit anchor (e.g. cold seed).

        Returns:
            IKResult containing joints_14 [B, 14] (grippers passthrough from
            seed) plus per-arm convergence flags and residuals.

        Notes:
            * pytorch_kinematics' PseudoInverseIK shares one retry_configs
              tensor across all problems in a batch, so this method effectively
              gives every chunk position the same warm-start. For per-position
              warm-start use `warm_solve_chunk_ik` which loops sequentially.
        """
        self._ensure_ik()

        if abs_cart_14.dim() != 2 or abs_cart_14.shape[1] != 14:
            raise ValueError(f"abs_cart_14 must be [B, 14], got {tuple(abs_cart_14.shape)}")
        B = abs_cart_14.shape[0]
        if seed_joints_14.dim() == 1:
            seed = seed_joints_14
        elif seed_joints_14.dim() == 2 and seed_joints_14.shape[0] == B:
            # Use the first row's joints as the shared seed; per-row seed is
            # supplied by warm_solve_chunk_ik (sequential).
            seed = seed_joints_14[0]
        else:
            raise ValueError(
                f"seed_joints_14 must be [14] or [B={B}, 14], got {tuple(seed_joints_14.shape)}")
        if seed.shape[-1] != 14:
            raise ValueError(f"seed_joints_14 last dim must be 14, got {seed.shape[-1]}")

        abs_cart_14 = abs_cart_14.to(device=self.device, dtype=self.dtype)
        seed = seed.to(device=self.device, dtype=self.dtype)
        if fallback_seed_joints_14 is None:
            fallback_seed = seed
        else:
            if fallback_seed_joints_14.shape[-1] != 14:
                raise ValueError(
                    f"fallback_seed_joints_14 last dim must be 14, got "
                    f"{fallback_seed_joints_14.shape[-1]}")
            fallback_seed = fallback_seed_joints_14.to(device=self.device, dtype=self.dtype)

        # Split into per-arm targets
        L_pos  = abs_cart_14[:, 0:3]
        L_quat = abs_cart_14[:, 3:7]
        R_pos  = abs_cart_14[:, 7:10]
        R_quat = abs_cart_14[:, 10:14]
        L_T = pk.Transform3d(matrix=_build_transform_4x4(L_pos, L_quat))
        R_T = pk.Transform3d(matrix=_build_transform_4x4(R_pos, R_quat))

        # Seed: PseudoInverseIK consumes `initial_config`. Shape [DOF] is
        # broadcast across all M problems and num_retries=1.
        self._left_ik.initial_config = seed[LEFT_ARM_IDX]
        self._right_ik.initial_config = seed[RIGHT_ARM_IDX]

        L_sol = self._left_ik.solve(L_T)   # solutions: [B, 1, 6]
        R_sol = self._right_ik.solve(R_T)

        L_joints = L_sol.solutions[:, 0, :]   # [B, 6]
        R_joints = R_sol.solutions[:, 0, :]

        # Assemble 14-D output; grippers come straight from the seed.
        joints_14 = torch.zeros(B, 14, device=self.device, dtype=self.dtype)
        joints_14[:, LEFT_ARM_IDX] = L_joints
        joints_14[:, LEFT_GRIPPER_IDX] = seed[LEFT_GRIPPER_IDX]
        joints_14[:, RIGHT_ARM_IDX] = R_joints
        joints_14[:, RIGHT_GRIPPER_IDX] = seed[RIGHT_GRIPPER_IDX]

        # Per-arm convergence flags. When IK didn't converge for an arm,
        # the LM solver's best-effort joints don't actually reach the
        # requested pose, so dispatching them would just be tracking noise.
        # Fall back to `fallback_seed` for those rows: by default this is the
        # warm-start seed, which inside warm_solve_chunk_ik is the previous
        # step's solved joints. The failing arm therefore holds at its last
        # committed value (smoothly resumes once a step converges again).
        # Each arm is independent; the other arm continues with its converged
        # IK. The convergence flags themselves are preserved verbatim in
        # IKResult so downstream safety gates can still see which targets
        # were unreachable.
        L_conv_b = (L_sol.converged_pos & L_sol.converged_rot)[:, 0]
        R_conv_b = (R_sol.converged_pos & R_sol.converged_rot)[:, 0]
        if (~L_conv_b).any():
            rows = (~L_conv_b).nonzero(as_tuple=True)[0]
            for c in LEFT_ARM_IDX:
                joints_14[rows, c] = fallback_seed[c]
        if (~R_conv_b).any():
            rows = (~R_conv_b).nonzero(as_tuple=True)[0]
            for c in RIGHT_ARM_IDX:
                joints_14[rows, c] = fallback_seed[c]

        # Diagnostics
        seed_full = seed.unsqueeze(0).expand(B, 14)
        delta_full = (joints_14 - seed_full).abs()                       # [B, 14]
        # only arm dims for "biggest joint move" check (grippers passthrough is 0)
        arm_idx = LEFT_ARM_IDX + RIGHT_ARM_IDX
        max_delta_from_seed = delta_full[:, arm_idx].max(dim=1).values   # [B]
        if B >= 2:
            step_deltas = (joints_14[1:, arm_idx] - joints_14[:-1, arm_idx]).abs()
            max_delta_in_chunk = step_deltas.max()
        else:
            max_delta_in_chunk = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        return IKResult(
            joints_14=joints_14,
            left_converged=(L_sol.converged_pos & L_sol.converged_rot)[:, 0].clone(),
            right_converged=(R_sol.converged_pos & R_sol.converged_rot)[:, 0].clone(),
            left_pos_err=L_sol.err_pos[:, 0].clone(),
            right_pos_err=R_sol.err_pos[:, 0].clone(),
            left_rot_err=L_sol.err_rot[:, 0].clone(),
            right_rot_err=R_sol.err_rot[:, 0].clone(),
            left_iterations=int(L_sol.iterations),
            right_iterations=int(R_sol.iterations),
            max_delta_from_seed=max_delta_from_seed,
            max_delta_in_chunk=max_delta_in_chunk,
        )

    @torch.no_grad()
    def warm_solve_chunk_ik(self, abs_cart_chunk_14: torch.Tensor,
                            seed_joints_14: torch.Tensor) -> IKResult:
        """Sequential per-step warm-started IK over an action chunk.

        For position t, seed = previous step's solution (t=0 seeded from the
        provided joint state). This avoids the all-share-one-seed limitation
        of `solve_ik` and produces a temporally smooth joint trajectory — the
        regime an impedance controller would actually execute.

        Args:
            abs_cart_chunk_14: [H, 14] absolute target poses (Lpos, Lquat, Rpos, Rquat).
            seed_joints_14: [14] current joint state.

        Returns:
            IKResult with stacked diagnostics across the H steps.
        """
        self._ensure_ik()
        H = abs_cart_chunk_14.shape[0]
        cur_seed = seed_joints_14
        joints_list, L_conv, R_conv, L_perr, R_perr, L_rerr, R_rerr = ([] for _ in range(7))
        L_iters_max, R_iters_max = 0, 0
        max_seed_deltas = []
        for t in range(H):
            r = self.solve_ik(abs_cart_chunk_14[t:t + 1], cur_seed)
            joints_list.append(r.joints_14[0])
            L_conv.append(r.left_converged[0])
            R_conv.append(r.right_converged[0])
            L_perr.append(r.left_pos_err[0])
            R_perr.append(r.right_pos_err[0])
            L_rerr.append(r.left_rot_err[0])
            R_rerr.append(r.right_rot_err[0])
            L_iters_max = max(L_iters_max, r.left_iterations)
            R_iters_max = max(R_iters_max, r.right_iterations)
            max_seed_deltas.append(r.max_delta_from_seed[0])
            cur_seed = r.joints_14[0]  # warm-start next step

        joints_14 = torch.stack(joints_list, dim=0)
        arm_idx = LEFT_ARM_IDX + RIGHT_ARM_IDX
        if H >= 2:
            step_deltas = (joints_14[1:, arm_idx] - joints_14[:-1, arm_idx]).abs()
            max_delta_in_chunk = step_deltas.max()
        else:
            max_delta_in_chunk = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        return IKResult(
            joints_14=joints_14,
            left_converged=torch.stack(L_conv),
            right_converged=torch.stack(R_conv),
            left_pos_err=torch.stack(L_perr),
            right_pos_err=torch.stack(R_perr),
            left_rot_err=torch.stack(L_rerr),
            right_rot_err=torch.stack(R_rerr),
            left_iterations=L_iters_max,
            right_iterations=R_iters_max,
            max_delta_from_seed=torch.stack(max_seed_deltas),
            max_delta_in_chunk=max_delta_in_chunk,
        )
