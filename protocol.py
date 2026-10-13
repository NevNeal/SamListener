#!/usr/bin/env python3
"""
The wire format between listen.py (the Mac) and server.py (the GPU node).
Numpy only, so both ends can import it.

Request    POST /segment
           Authorization: Bearer <token>
           X-Sam-Params:  JSON, see PARAMS_HEADER
           body:          one JPEG frame

Response   4 bytes   big-endian length of the JSON that follows
           JSON      {"shape": [N, H, W], "scores": [...], "boxes": [...], ...}
           the rest  zlib( np.packbits( N x H x W stack of 0/1 ) )

The mask payload is the same bit-packing remote_sam3's maskio.py uses, so a
snapshot saved by listen.py loads with maskio.load_masks().
"""

import json
import struct
import zlib

import numpy as np

# JSON object carried in this request header:
#   prompt          str    the text prompt ("" for boxes only)
#   threshold       float  keep detections at or above this score
#   mask_threshold  float  binarisation cutoff inside SAM3's post-processing
#   boxes           list   [x1, y1, x2, y2, label] in pixels of the JPEG sent;
#                          label 1 = "things like this", 0 = "not this"
PARAMS_HEADER = "X-Sam-Params"

DEFAULT_PORT = 8766


def encode_result(masks, scores, meta=None):
    """masks: N x H x W array of 0/1 (N may be 0). Returns the response body."""
    masks = np.asarray(masks).astype(bool)
    if masks.ndim != 3:
        raise ValueError(f"masks must be N x H x W, got shape {masks.shape}")
    header = dict(meta or {})
    header["shape"] = [int(v) for v in masks.shape]
    header["scores"] = [round(float(s), 4) for s in scores]
    head = json.dumps(header).encode()
    packed = np.packbits(masks, axis=None).tobytes()
    return struct.pack(">I", len(head)) + head + zlib.compress(packed, 1)


def decode_result(body):
    """The response body -> (masks as an N x H x W bool array, scores, meta)."""
    (head_len,) = struct.unpack(">I", body[:4])
    meta = json.loads(body[4:4 + head_len])
    shape = tuple(meta["shape"])
    count = int(np.prod(shape))
    packed = np.frombuffer(zlib.decompress(body[4 + head_len:]), dtype=np.uint8)
    masks = np.unpackbits(packed, count=count).reshape(shape).astype(bool)
    return masks, [float(s) for s in meta["scores"]], meta
