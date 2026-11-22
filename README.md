# ComfyUI-SharpestFrames

A tiny ComfyUI node that picks the **sharpest frames of a video** around target frame numbers.

Made for character sheets built from an orbit video (MiniMax H3, Wan, LTX…): fixed frame numbers sometimes land on a frame
blurred by camera motion or a focus pull. This node searches `± window` frames around each target and keeps the sharpest one
(variance of the Laplacian, measured on the central area so the flat background does not count).

## Nodes

- **Sharpest Frames (blur detector)** — `images` (IMAGE batch), `frames` (`"1,30,54,90,100,112,123"`, 1-based), `window` (default 6),
  `center_crop` (default 0.3), optional `mode` (`around targets` / `spread over video` = N equal segments, max variety) and `min_gap` (default 4, no two tiles from the same moment) → `images` (one sharpest frame per target, same order) + `chosen_frames` (STRING).
- **Image From Batch (index)** — one image of a batch by 0-based index, handy to split the picked frames into sheet tiles.

No dependencies beyond ComfyUI (PyTorch). Install: clone into `ComfyUI/custom_nodes/` and restart ComfyUI.

```
cd ComfyUI/custom_nodes
git clone https://github.com/nlienard/ComfyUI-SharpestFrames
```

## License

MIT
