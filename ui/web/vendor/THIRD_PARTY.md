# Three.js browser dependencies

Three.js version **0.180.0** (MIT), from the official npm `three` package.
The existing `three.core.js` declares revision `180`; addon sources match it.

`addons/` contains unmodified official `examples/jsm/` modules: RGBELoader and
HDRLoader, EffectComposer, RenderPass, UnrealBloomPass, OutputPass, and their
relative shader/pass dependencies. Relative directory structure is preserved.
The existing renderer modules and OrbitControls use the same Three.js license.

Source: https://github.com/mrdoob/three.js/tree/r180/examples/jsm
Documentation: https://threejs.org/docs/pages/UnrealBloomPass.html and
https://threejs.org/manual/pages/post-processing.html

Environment lighting uses the existing Poly Haven CC0 asset
`assets/hdri_polyhaven/kloofendal_48d_partly_cloudy_puresky_1k.hdr`.
See `assets/LICENSES.md` for its source and license.

## MIT license

The MIT License

Copyright © 2010-2025 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
