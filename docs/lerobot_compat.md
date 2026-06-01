# dk1 LeRobot ↔ Cosmos loader compatibility

## dk1 video format (non-standard: one video file per episode + from/to timestamps)

- `info.json` `video_path`: `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`
- **One mp4 per episode per camera** (file_index unique per episode; ep0→file0, ep1→file1, …).
- episodes-meta (`meta/episodes/chunk-*/file-*.parquet`) per camera columns:
  `videos/{key}/chunk_index`, `videos/{key}/file_index`,
  `videos/{key}/from_timestamp`, `videos/{key}/to_timestamp`.
- Per-frame `timestamp` (in data parquet) is **episode-relative** (starts at 0).
- Example (swan ep0): `from_ts=1.000, to_ts=16.650`; per-frame timestamp ∈ [0.000, 15.633].
  Absolute video time = `from_ts + timestamp` ∈ [1.0, 16.633] ⊂ [from_ts, to_ts]. The 1.0s
  offset is a per-episode-video lead-in.

## Cosmos `DROIDLeRobotDataset` handling (data/vfm/action/datasets/droid_lerobot_dataset.py)

- `_video_path(episode, video_key)` reads `videos/{key}/chunk_index` + `videos/{key}/file_index`
  (primary keys; fallbacks to episode_chunk/data_*) → resolves per-episode file correctly.
- `_load_concat_video` decodes at `from_timestamp + per-frame timestamp` via
  `lerobot.datasets.video_utils.decode_video_frames`. Does NOT use `to_timestamp`.

## Conclusion

**The per-episode-video + from/to-timestamp scheme is already supported** — no patch to the
video/timestamp mechanism is required. Cosmos's `from_timestamp + timestamp` correctly addresses
dk1's offset per-episode videos.

The dk1-specific changes belong in our `DK1LeRobotDataset` (subclass/rewrite), not the video logic:
1. `_IMAGE_FEATURES`: `observation.images.{head,left_wrist,right_wrist}` (vs DROID's wrist/exterior keys).
2. `_STATE_FEATURE`: `observation.state` (14D) vs DROID `observation.state.cartesian_position`.
3. View/concat layout for the 3 dk1 cams.
4. `_build_raw_action`: 14D bimanual joint-space (vs DROID 10D cartesian delta).
5. Tune `tolerance_s` if needed; confirm codec (AV1?) decodes via torchcodec.

## CONFIRMED empirically (2026-06-01, cosmos venv)

`decode_video_frames("…/observation.images.head/chunk-000/file-000.mp4", [from_ts(1.0)+ts...], 2e-4)`
→ `(8, 3, 360, 640)` float32, valid range. The per-episode video + `from_ts` offset decodes correctly
through Cosmos's torchcodec path. **No patch to the video/timestamp mechanism required.**
`build_action_spec(Joint(6,"left_arm"), Gripper("left"), Joint(6,"right_arm"), Gripper("right"))` → 14D.

## ENV NOTE (required)

torchcodec needs the cu13 NVIDIA libs on the linker path, else `libnppicc.so.13` fails to load:
```
export LD_LIBRARY_PATH=/workspace/code/cosmos-framework/.venv/lib/python3.13/site-packages/nvidia/cu13/lib
```
(Same class of issue as FastWAM's cu13 LD_LIBRARY_PATH requirement. The README's `LD_LIBRARY_PATH=`
empty-export is wrong for video decoding on this box.)
