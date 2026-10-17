# Realistic mining robots demo

**Capture in progress.** This document will be finalized from the completed
recording and its measured episode report; no final video is claimed yet.

## What changed

- A visibly articulated Universal Robots UR5e collector with silver links,
  blue joint covers, metallic tooling and parallel jaws.
- Industrial mobile chassis with chamfered covers, vents, fasteners and wheel
  hubs; a separate tracked drilling rig.
- Enclosed faceted rock, mine supports and service pipes, improved illumination
  and closer camera framing that keeps both robots and all three carts visible.
- Actual robot motion, RGB-D sensor feed, measured drill thrust/torque/depth,
  gripper contact forces and cart loading/dispatch labels.

Presentation geometry is applied to a separate rendering model. The policy
camera and locked benchmark physics are not replaced by aesthetic geometry.

## Run

Interactive native view, with a display available:

```bash
.venv/bin/python main.py run-mine --seed 101
```

GPU0 recording on the configured machine:

```bash
ssh cis-a6000
cd ~/deepcore-n1
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
__EGL_VENDOR_LIBRARY_FILENAMES="$HOME/.local/share/glvnd/egl_vendor.d/10_nvidia.json" \
~/perception-loop-n1/.venv/bin/python main.py run-mine --seed 101 --out dev/native_realistic_fleet.mp4
```

## Scope and limitations

This is a simulation demonstration, not physical robot validation. The fleet
uses actuated planar bases, discrete contact-work rock removal and a
contact-conditioned weld grasp. The collector follows corrupted RGB-D tracks
and its own joint/base encoders. Render/evaluation state is not a policy input.
The UR5e assets come from MuJoCo Menagerie; their license is retained under
`assets/robot_reference/universal_robots_ur5e/`.

The synthetic enclosure is presentation scenery, not a claim of measured CMU
survey reconstruction. A completed visual demo does not establish fleet policy
learning, unseen-seed generalization, or real-time 60 FPS rendering.
