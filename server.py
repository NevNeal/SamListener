#!/usr/bin/env python3
"""
The GPU end of SamListener: holds SAM3 in memory and segments whatever frame
it is sent, with whatever prompt came along with that frame.

    python server.py                 # on a GPU node; slurm/listener.sbatch does this
    python server.py --mock          # no torch, no GPU: fake masks, for testing

Every request carries its own prompt, threshold and exemplar boxes, so changing
the prompt needs nothing here: the next frame simply arrives with new text.
Nothing is kept between frames and nothing is written to disk.

Standard library plus what remote_sam3's conda env already has (torch,
transformers, pillow, numpy). protocol.py describes the wire format.
"""

import argparse
import json
import os
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import numpy as np
from PIL import Image

from protocol import DEFAULT_PORT, PARAMS_HEADER, encode_result

MAX_FRAME_BYTES = 20 * 1024 * 1024
MAX_MASKS = 200              # most confident first; the rest are dropped


class Segmenter:
    """SAM3, loaded once. One forward pass per call, guarded by a lock because
    the HTTP server is threaded and there is one GPU."""

    def __init__(self, bf16):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.bf16 = bf16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            props = torch.cuda.get_device_properties(0)
            self.gpu = f"{props.name} ({props.total_memory / 1e9:.0f} GB)"
        else:
            self.gpu = "cpu"
            print("gpu        : NONE - running on CPU, this will be very slow")
        t0 = time.perf_counter()
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        print(f"gpu        : {self.gpu}")
        print(f"model load : {time.perf_counter() - t0:.1f}s")
        self.lock = threading.Lock()

    def _inputs(self, image, prompt, boxes):
        kwargs = {"return_tensors": "pt"}
        if boxes:
            kwargs["input_boxes"] = [[[float(v) for v in b[:4]] for b in boxes]]
            kwargs["input_boxes_labels"] = [[int(b[4]) for b in boxes]]
        if not prompt:
            return self.processor(images=image, **kwargs)
        # SAM3's processor wants one text per image; a build that instead
        # broadcasts a single string rejects the list, so try the list first.
        try:
            return self.processor(images=[image], text=[prompt], **kwargs)
        except (TypeError, ValueError):
            return self.processor(images=image, text=prompt, **kwargs)

    def __call__(self, image, prompt, boxes, threshold, mask_threshold):
        torch = self.torch
        cuda = self.device == "cuda"
        with self.lock:
            inputs = self._inputs(image, prompt, boxes).to(self.device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                if self.bf16 and cuda:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
            if cuda:
                torch.cuda.synchronize()    # kernels are async; wait so this is real GPU time
            infer_ms = (time.perf_counter() - t0) * 1000
            detection = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]
        masks, scores, found = _keep(detection, image.size, threshold)
        return masks, scores, found, infer_ms


def _keep(detection, size, threshold):
    """SAM3's raw output for one image -> (N x H x W bool, scores, boxes), most
    confident first."""
    width, height = size
    raw_masks, raw_scores = detection.get("masks"), detection.get("scores")
    raw_boxes = detection.get("boxes")
    kept = []
    if raw_masks is not None and raw_scores is not None and len(raw_scores):
        scores = [float(s) for s in raw_scores]
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        for i in order[:MAX_MASKS]:
            if scores[i] < threshold:
                continue
            mask = np.squeeze(raw_masks[i].detach().float().cpu().numpy()) > 0
            if mask.shape != (height, width):
                resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)
                mask = np.asarray(resized) > 0
            if not mask.any():
                continue
            box = [round(float(v), 1) for v in raw_boxes[i]] if raw_boxes is not None else None
            kept.append((mask, scores[i], box))
    if not kept:
        return np.zeros((0, height, width), dtype=bool), [], []
    return (np.stack([k[0] for k in kept]), [k[1] for k in kept], [k[2] for k in kept])


class MockSegmenter:
    """Stands in for SAM3 where there is no GPU. Returns the darker and the
    brighter regions of the frame as two 'detections', plus the inside of every
    positive exemplar box, so the viewer has real pixels to align and draw."""

    gpu = "mock (no model)"

    def __call__(self, image, prompt, boxes, threshold, mask_threshold):
        t0 = time.perf_counter()
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        spread = gray.std() or 1.0
        found = []
        if prompt:
            found.append((gray < gray.mean() - 0.5 * spread, 0.90))
            found.append((gray > gray.mean() + 0.5 * spread, 0.60))
        for x1, y1, x2, y2, label in boxes:
            if label:
                inside = np.zeros(gray.shape, dtype=bool)
                inside[int(y1):int(y2), int(x1):int(x2)] = True
                found.append((inside, 0.75))
        found = [(m, s) for m, s in found if s >= threshold and m.any()]
        found.sort(key=lambda pair: pair[1], reverse=True)
        if not found:
            masks = np.zeros((0, *gray.shape), dtype=bool)
        else:
            masks = np.stack([m for m, _ in found])
        infer_ms = (time.perf_counter() - t0) * 1000
        return masks, [s for _, s in found], [None] * len(found), infer_ms


class Stats:
    """A line in the log every so often, instead of one per frame."""

    def __init__(self, every):
        self.every = every
        self.lock = threading.Lock()
        self.count, self.infer_ms, self.masks = 0, 0.0, 0
        self.prompt = ""
        self.since = time.monotonic()

    def add(self, prompt, infer_ms, n_masks):
        with self.lock:
            self.count += 1
            self.infer_ms += infer_ms
            self.masks += n_masks
            self.prompt = prompt
            elapsed = time.monotonic() - self.since
            if elapsed < self.every:
                return
            print(f"{time.strftime('%H:%M:%S')}  {self.count / elapsed:5.2f} frames/s  "
                  f"gpu {self.infer_ms / self.count:6.0f} ms/frame  "
                  f"{self.masks / self.count:5.1f} masks/frame  prompt '{self.prompt}'",
                  flush=True)
            self.count, self.infer_ms, self.masks = 0, 0.0, 0
            self.since = time.monotonic()


def make_handler(segmenter, token, stats):

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"       # keep-alive: one connection for the whole stream

        def log_message(self, *args):       # Stats does the logging
            pass

        def _send(self, status, body, content_type="application/octet-stream"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, payload):
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _authorized(self):
            given = self.headers.get("Authorization", "")
            if secrets.compare_digest(given, f"Bearer {token}"):
                return True
            # The body was never read, so this connection cannot be reused.
            self.close_connection = True
            self._json(401, {"error": "bad or missing token"})
            return False

        def do_GET(self):
            if not self._authorized():
                return
            if self.path == "/health":
                self._json(200, {"ok": True, "gpu": segmenter.gpu, "host": socket.gethostname()})
            else:
                self._json(404, {"error": "no such path"})

        def do_POST(self):
            if not self._authorized():
                return
            length = int(self.headers.get("Content-Length") or 0)
            if self.path != "/segment" or not 0 < length <= MAX_FRAME_BYTES:
                self.close_connection = True
                self._json(400, {"error": "POST one JPEG frame to /segment"})
                return
            started = time.perf_counter()
            body = self.rfile.read(length)
            try:
                params = json.loads(self.headers.get(PARAMS_HEADER) or "{}")
                prompt = str(params.get("prompt") or "").strip()
                boxes = [[float(v) for v in b[:4]] + [int(b[4])] for b in params.get("boxes") or []]
                threshold = float(params.get("threshold", 0.5))
                mask_threshold = float(params.get("mask_threshold", 0.5))
                image = Image.open(BytesIO(body)).convert("RGB")
            except Exception as error:
                self._json(400, {"error": f"bad request: {error}"})
                return
            try:
                if prompt or boxes:
                    masks, scores, found, infer_ms = segmenter(
                        image, prompt, boxes, threshold, mask_threshold)
                else:
                    masks = np.zeros((0, image.height, image.width), dtype=bool)
                    scores, found, infer_ms = [], [], 0.0
            except Exception as error:
                # Most likely out of memory on a frame with very many objects.
                torch = getattr(segmenter, "torch", None)
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"ERROR segmenting ('{prompt}', {len(boxes)} boxes): "
                      f"{type(error).__name__}: {error}", flush=True)
                self._json(500, {"error": f"{type(error).__name__}: {error}"})
                return
            stats.add(prompt, infer_ms, len(scores))
            self._send(200, encode_result(masks, scores, {
                "boxes": found,
                "infer_ms": round(infer_ms, 1),
                "server_ms": round((time.perf_counter() - started) * 1000, 1),
            }))

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="0.0.0.0",
                        help="interface to listen on (default: all, so the login node can reach it)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--token", default=os.environ.get("SAM_LISTENER_TOKEN"),
                        help="shared secret; a random one is made and printed if not given")
    parser.add_argument("--bf16", action="store_true",
                        help="run the forward pass in bfloat16: faster, slightly different mask edges")
    parser.add_argument("--mock", action="store_true",
                        help="no model, fake masks: test the plumbing without a GPU")
    parser.add_argument("--login-host", default="hipergator",
                        help="the ssh alias printed in the tunnel command")
    parser.add_argument("--log-every", type=float, default=30,
                        help="seconds between throughput lines in the log")
    return parser.parse_args()


def main():
    args = parse_args()
    token = args.token or secrets.token_urlsafe(12)
    segmenter = MockSegmenter() if args.mock else Segmenter(args.bf16)
    stats = Stats(args.log_every)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(segmenter, token, stats))
    server.daemon_threads = True

    node = socket.gethostname()
    print(f"listening  : {node}:{args.port}")
    print()
    print("On the Mac, in two terminals:")
    print()
    print(f"    ssh -N -L {args.port}:{node}:{args.port} {args.login_host}")
    print(f"    .venv/bin/python listen.py --port {args.port} --token {token}")
    print(flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
