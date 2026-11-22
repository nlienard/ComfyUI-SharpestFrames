"""ComfyUI-SharpestFrames: pick the sharpest frames of a video around target frame numbers.

Typical use: character sheets assembled from an orbit video (MiniMax H3, Wan, LTX...). Fixed frame numbers sometimes land
on a frame blurred by camera motion or a focus pull; this node looks +/- `window` frames around each target and keeps the
sharpest one (variance of the Laplacian on the central area of the frame).
"""
import torch
import torch.nn.functional as F

_LAPLACE = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)


def _sharpness(frames, crop):
    """frames: (N, H, W, C) in 0..1 -> (N,) variance of the Laplacian of the luma, on the central area."""
    g = frames[..., :3].mean(-1, keepdim=True).permute(0, 3, 1, 2)  # N,1,H,W
    _, _, h, w = g.shape
    if crop > 0:
        dy, dx = int(h * crop / 2), int(w * crop / 2)
        g = g[:, :, dy:h - dy, dx:w - dx]
    lap = F.conv2d(g, _LAPLACE.to(g.device, g.dtype))
    return lap.flatten(1).var(dim=1)


class SharpestFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "All the frames of the video (IMAGE batch)."}),
                "frames": ("STRING", {"default": "1,30,54,90,100,112,123",
                                      "tooltip": "Target frame numbers, comma separated (1 = first frame)."}),
                "window": ("INT", {"default": 6, "min": 0, "max": 100,
                                   "tooltip": "Search +/- this many frames around each target."}),
                "center_crop": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
                                          "tooltip": "Fraction of the frame border ignored when measuring sharpness (keeps the subject, drops the flat background)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "chosen_frames")
    OUTPUT_TOOLTIPS = ("The sharpest frame found around each target, in the same order.", "Chosen frame numbers (1-based), comma separated.")
    FUNCTION = "pick"
    CATEGORY = "image/video"
    DESCRIPTION = "Picks, around each target frame number, the sharpest frame of the video (variance of the Laplacian)."

    def pick(self, images, frames, window, center_crop):
        n = images.shape[0]
        targets = [int(x) for x in str(frames).replace(";", ",").split(",") if x.strip()]
        chosen = []
        for t in targets:
            t = min(max(t - 1, 0), n - 1)
            lo, hi = max(0, t - window), min(n, t + window + 1)
            scores = _sharpness(images[lo:hi].float(), center_crop)
            chosen.append(lo + int(torch.argmax(scores)))
        out = images[chosen]
        return (out, ",".join(str(c + 1) for c in chosen))


class ImageFromBatchIndex:
    """Small helper: one image of a batch by index (0-based), to split the picked frames into separate sheet tiles."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "index": ("INT", {"default": 0, "min": 0, "max": 9999})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get"
    CATEGORY = "image/video"

    def get(self, images, index):
        return (images[min(index, images.shape[0] - 1)].unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"SharpestFrames": SharpestFrames, "ImageFromBatchIndex": ImageFromBatchIndex}
NODE_DISPLAY_NAME_MAPPINGS = {"SharpestFrames": "Sharpest Frames (blur detector)", "ImageFromBatchIndex": "Image From Batch (index)"}
