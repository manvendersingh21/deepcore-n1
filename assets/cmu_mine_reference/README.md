# CMU 3D mine map references

Downloaded from the files linked by the [CMU 3D mine map page](https://www.cs.cmu.edu/~thrun/3D/mines/html/map0.html) on 2026-10-17 (completed by about 21:23 UTC). Total downloaded size: 132,758,969 bytes. The supplied `map0.html` was already in this directory and was not downloaded in this task.

| Local file | Source | Size | Format / validation |
|---|---|---:|---|
| [mine1-mod.vrml](./mine1-mod.vrml) | [CMU source](https://www.cs.cmu.edu/~thrun/3D/mines/mine-mapping/mine1-mod.vrml) | 32,808,388 B | VRML 1.0 ASCII (`#VRML V1.0 ascii`) |
| [bruceton1126.mpg](./bruceton1126.mpg) | [CMU source](https://www.cs.cmu.edu/~thrun/3D/mines/groundhog/anim/bruceton1126.mpg) | 31,825,967 B | MPEG-1 video, 320×240; `ffprobe` read duration 221.398 s |
| [mine1-anim2.avi](./mine1-anim2.avi) | [CMU source](https://www.cs.cmu.edu/~thrun/3D/mines/mine-mapping/mine1-anim2.avi) | 57,704,866 B | RIFF/AVI header. `ffprobe` rejects the stream metadata (`Invalid sample rate: -16246711`). A second GET of the same URL returned the same size and SHA-256; the original file is retained unchanged. |
| [mine1-anim-alt.avi](./mine1-anim-alt.avi) | [CMU source](https://www.cs.cmu.edu/~thrun/3D/mines/mine-mapping/mine1-anim-alt.avi) | 10,419,748 B | AVI, Microsoft MPEG-4 v2, 640×480, 30 fps; `ffprobe` read duration 27.567 s |

**License:** the linked CMU page gives no license statement for these files. Their legal status is unspecified here; this archive makes no claim that they are open source or may be redistributed.

`manifest.json` records the HTTP results, byte counts, file-format checks, and SHA-256 hashes. The large video files were inspected by header/metadata only; no video playback scripts or VRML programs were run.
