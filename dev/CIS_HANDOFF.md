# HIGHEST PRIORITY — DeepCore-N1 (cis-a6000)

This task is the highest priority on this machine. Our work takes precedence over anything else queued here.
Use GPU0 only (MUJOCO_EGL_DEVICE_ID=0). GPU1 belongs to another job: do not touch it.

## Headless rendering recipe (verified)
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export __EGL_VENDOR_LIBRARY_FILENAMES=$HOME/.local/share/glvnd/egl_vendor.d/10_nvidia.json
Use the venv at ~/perception-loop-n1/.venv/bin/python (mujoco 3.16). Verified: Episode(11).observe() returns (480,640,3) in 0.15 s.

## Job queue (after the DeepCore-N1 repo is pushed here)
1. Seed sweeps: evaluate workspace/candidate_pipeline.py on seeds 11..33 and 101..103 and report aggregates.
2. Report results as JSON only. Do not write to shared_brain/ or workspace/ on this machine without an instruction.
