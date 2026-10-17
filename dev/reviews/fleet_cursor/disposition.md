# Peer A disposition: native mining UI review

Root reviewed both confirmed findings. Static inspection of the current source shows both presentation fixes in place:

| Finding | Root disposition | Current status |
|---|---|---|
| Fleet debris/cart label was anchored away from the carts | Resolved in current code. `scene.js` creates each cart label from its `cart_N_floor` mesh, so the label follows the cart pose; the per-frame text shows load and dispatch status. | Static source checked; browser/runtime behavior was not rerun. |
| Live scrubber max stayed at 219 (~11 s) | Resolved in current code. `app.js` sets the live range max from `(frame.duration || 11) * 20` while rendering live frames. | Static source checked; browser/runtime behavior was not rerun. |

No remaining code change is assigned from these findings. The HACP report remains the independent static review; this disposition reflects the root-reviewed current source, not a runtime verification.
