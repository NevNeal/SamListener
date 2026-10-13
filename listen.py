#!/usr/bin/env python3
"""
The Mac end of SamListener: shows the capture card live, sends frames to the
SAM3 server, and draws the masks that come back over the video.

    .venv/bin/python listen.py --token <token printed by the server>
    .venv/bin/python listen.py --list            # which camera index is the card?
    .venv/bin/python listen.py --source clip.mp4 # a video or image instead of the card

Keys, with the video window focused:

    p or Tab   edit the prompt     (type, Enter to apply, Esc to cancel)
    1 .. 9     jump to a preset    (--prompts "cell wall,nucleus,chloroplast")
    [  ]       score threshold down / up
    drag       box an example: "segment things like this"
    shift-drag box a counter-example: "not this"     (right-drag does the same)
    c          clear the boxes
    space      freeze the frame    (to box something on a moving specimen)
    m          masks on / off
    r          start / stop recording the raw video (no overlay) to recordings/
    s          snapshot: frame + masks + prompt to snapshots/
    n          next camera
    q or Esc   quit

Only the newest frame is ever sent: while the server is busy, frames are shown
but not queued, so the masks trail the video by one round trip and never more.
"""

import argparse
import http.client
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from protocol import DEFAULT_PORT, PARAMS_HEADER, decode_result

WINDOW = "SamListener"
HUD_HEIGHT = 84
MASK_ALPHA = 0.40
MIN_BOX = 8                  # pixels; a smaller drag is a stray click
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# BGR. Bright and saturated, because micrographs are mostly pale or green.
PALETTE = [
    (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 165, 255), (0, 255, 128),
    (255, 128, 0), (128, 0, 255), (0, 0, 255), (255, 0, 128), (128, 255, 255),
]
POSITIVE, NEGATIVE = (0, 220, 0), (0, 0, 230)


# -- The video source ----------------------------------------------------------

class Camera(threading.Thread):
    """Reads the source as fast as it delivers and keeps only the newest frame.
    Also owns the recording, so no frame is dropped when the display is slow."""

    def __init__(self, args):
        super().__init__(daemon=True)
        self.args = args
        self.device = args.device
        self.lock = threading.Lock()
        self.frame, self.seq = None, 0
        self.status = "opening"
        self.fps = 0.0
        self.writer, self.recording_to = None, None
        self.reopen = threading.Event()
        self.stop = threading.Event()

    def latest(self):
        with self.lock:
            return self.seq, self.frame

    def next_device(self):
        self.device = (self.device + 1) % self.args.max_devices
        self.reopen.set()

    def toggle_recording(self):
        with self.lock:
            if self.writer is not None:
                self.writer.release()
                self.writer, done = None, self.recording_to
                self.recording_to = None
                return f"saved {done}"
            if self.frame is None:
                return "nothing to record yet"
            height, width = self.frame.shape[:2]
            folder = Path(self.args.out) / "recordings"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{stamp()}.mp4"
            fps = self.fps if 1 <= self.fps <= 120 else 30
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))
            if not writer.isOpened():
                return "could not open a video writer"
            self.writer, self.recording_to = writer, path
            return f"recording to {path}"

    def _publish(self, frame):
        with self.lock:
            self.frame = frame
            self.seq += 1
            if self.writer is not None:
                self.writer.write(frame)

    def _open(self):
        source = self.args.source
        if source:
            capture = cv2.VideoCapture(str(source))
            name = Path(source).name
        else:
            capture = cv2.VideoCapture(self.device, cv2.CAP_AVFOUNDATION)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
            capture.set(cv2.CAP_PROP_FPS, self.args.fps)
            name = f"camera {self.device}"
        return capture, name

    def run(self):
        source = self.args.source
        if source and Path(source).suffix.lower() in IMAGE_EXTS:
            still = cv2.imread(str(source))
            if still is None:
                self.status = f"cannot read {source}"
                return
            self.status = f"{Path(source).name} {still.shape[1]}x{still.shape[0]}"
            self._publish(still)        # once: a still only needs segmenting once per prompt
            self.stop.wait()
            return

        while not self.stop.is_set():
            self.reopen.clear()
            capture, name = self._open()
            if not capture.isOpened():
                self.status = f"{name}: not available"
                capture.release()
                self.reopen.wait(2)
                continue
            file_fps = capture.get(cv2.CAP_PROP_FPS) if source else 0
            count, since = 0, time.monotonic()
            while not self.stop.is_set() and not self.reopen.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    if source:          # end of the clip: loop it
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.status = f"{name}: no frames"
                    break
                self._publish(frame)
                count += 1
                elapsed = time.monotonic() - since
                if elapsed >= 1:
                    self.fps, count, since = count / elapsed, 0, time.monotonic()
                self.status = (f"{name} {frame.shape[1]}x{frame.shape[0]} "
                               f"{self.fps:.0f}fps")
                if source and file_fps > 0:
                    time.sleep(1 / file_fps)
            capture.release()
            if not self.reopen.is_set():
                self.stop.wait(1)
        with self.lock:
            if self.writer is not None:
                self.writer.release()
                self.writer = None


def list_devices(args):
    """Open each camera index in turn and say what it delivers. Opening a camera
    turns it on, so the webcam light and a nearby iPhone will both wake up."""
    folder = Path(args.out) / "devices"
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(args.max_devices):
        capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not capture.isOpened():
            print(f"camera {index}: not available")
            capture.release()
            continue
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        frame = None
        for _ in range(10):             # the first frames are often black
            ok, grabbed = capture.read()
            if ok:
                frame = grabbed
        capture.release()
        if frame is None:
            print(f"camera {index}: opens, but delivers no frames")
            continue
        path = folder / f"camera_{index}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"camera {index}: {frame.shape[1]}x{frame.shape[0]}  "
              f"brightness {frame.mean():.0f}/255  -> {path}")
    print(f"\nLook at the pictures in {folder}, then run with --device N.")


# -- What the user has asked for -----------------------------------------------

class State:
    """The prompt, threshold and boxes, shared between the window and the
    sender. Boxes are stored as fractions of the frame so they survive a change
    of camera or of the size sent."""

    def __init__(self, args):
        self.lock = threading.Lock()
        self.prompt = args.prompt
        self.threshold = args.threshold
        self.mask_threshold = args.mask_threshold
        self.boxes = []             # (x1, y1, x2, y2, label), all 0..1
        self.frozen = None          # (seq, frame) while frozen

    def params(self):
        with self.lock:
            return self.prompt, round(self.threshold, 3), self.mask_threshold, tuple(self.boxes)


class Result:
    def __init__(self, seq, frame, masks, scores, meta, prompt, threshold, round_trip_ms):
        self.seq, self.frame = seq, frame
        self.masks, self.scores, self.meta = masks, scores, meta
        self.prompt, self.threshold = prompt, threshold
        self.round_trip_ms = round_trip_ms
        self.layer = None           # built by the window, cached here


# -- Talking to the server -----------------------------------------------------

class Sender(threading.Thread):
    """Sends the newest frame, waits for its masks, repeats. A frame is sent
    again only if it or the prompt has changed, so a still or a frozen frame
    costs one request per prompt rather than a stream of identical ones."""

    def __init__(self, args, camera, state):
        super().__init__(daemon=True)
        self.args, self.camera, self.state = args, camera, state
        self.lock = threading.Lock()
        self.result = None
        self.results = 0
        self.status = "connecting"
        self.rate = 0.0
        self.conn = None
        self.stop = threading.Event()

    def latest(self):
        with self.lock:
            return self.result

    def _headers(self, extra=None):
        headers = {"Authorization": f"Bearer {self.args.token or ''}"}
        headers.update(extra or {})
        return headers

    def _request(self, method, path, body=None, headers=None):
        if self.conn is None:
            self.conn = http.client.HTTPConnection(self.args.host, self.args.port,
                                                   timeout=self.args.timeout)
        self.conn.request(method, path, body=body, headers=self._headers(headers))
        response = self.conn.getresponse()
        data = response.read()
        if response.status != 200:
            try:
                message = json.loads(data).get("error", "")
            except ValueError:
                message = data[:120].decode(errors="replace")
            raise RuntimeError(f"{response.status} {message}")
        return data

    def _drop(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def run(self):
        last_key, healthy = None, False
        done, since = 0, time.monotonic()
        while not self.stop.is_set():
            try:
                if not healthy:
                    info = json.loads(self._request("GET", "/health"))
                    self.status = f"{info.get('gpu', 'ok')} on {info.get('host', '?')}"
                    self.gpu = self.status
                    healthy = True

                seq, frame = self.state.frozen or self.camera.latest()
                prompt, threshold, mask_threshold, boxes = self.state.params()
                key = (seq, prompt, threshold, mask_threshold, boxes)
                if frame is None or key == last_key:
                    time.sleep(0.004)
                    continue

                started = time.perf_counter()
                small = shrink(frame, self.args.send_size)
                if prompt or boxes:
                    height, width = small.shape[:2]
                    ok, jpeg = cv2.imencode(".jpg", small,
                                            [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
                    if not ok:
                        raise RuntimeError("could not encode the frame")
                    params = {
                        "prompt": prompt, "threshold": threshold,
                        "mask_threshold": mask_threshold,
                        "boxes": [[x1 * width, y1 * height, x2 * width, y2 * height, label]
                                  for x1, y1, x2, y2, label in boxes],
                    }
                    body = self._request("POST", "/segment", jpeg.tobytes(), {
                        "Content-Type": "image/jpeg", PARAMS_HEADER: json.dumps(params)})
                    masks, scores, meta = decode_result(body)
                else:
                    masks = np.zeros((0, *small.shape[:2]), dtype=bool)
                    scores, meta = [], {}
                round_trip = (time.perf_counter() - started) * 1000
                with self.lock:
                    self.result = Result(seq, frame, masks, scores, meta,
                                         prompt, threshold, round_trip)
                    self.results += 1
                last_key = key
                self.status = self.gpu

                done += 1
                elapsed = time.monotonic() - since
                if elapsed >= 2:
                    self.rate, done, since = done / elapsed, 0, time.monotonic()
            except Exception as error:
                self._drop()
                healthy = False
                self.rate = 0.0
                if isinstance(error, ConnectionRefusedError):
                    self.status = f"no server on {self.args.host}:{self.args.port}"
                else:
                    self.status = f"{type(error).__name__}: {error}"[:90]
                self.stop.wait(1.5)
        self._drop()


def shrink(frame, long_side):
    """The frame with its longer side reduced to long_side. Never enlarged."""
    height, width = frame.shape[:2]
    scale = long_side / max(height, width)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (round(width * scale), round(height * scale)),
                      interpolation=cv2.INTER_AREA)


# -- Drawing -------------------------------------------------------------------

class Layer:
    """One result's masks, painted once at display size: where to tint, and the
    outlines. Blending it onto each new video frame is then two array copies."""

    def __init__(self, masks, width, height):
        self.size = (width, height)
        self.color = np.zeros((height, width, 3), dtype=np.uint8)
        self.fill = np.zeros((height, width), dtype=bool)
        edges = np.zeros((height, width, 3), dtype=np.uint8)
        # Least confident first, so the most confident mask ends up on top.
        for index in range(len(masks) - 1, -1, -1):
            color = PALETTE[index % len(PALETTE)]
            mask = cv2.resize(masks[index].astype(np.uint8), (width, height),
                              interpolation=cv2.INTER_NEAREST)
            inside = mask > 0
            self.color[inside] = color
            self.fill |= inside
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(edges, contours, -1, color, max(1, round(width / 900)))
        self.edge = edges.any(axis=2)
        self.edge_color = edges

    def draw(self, frame):
        out = frame.copy()
        if self.fill.any():
            tinted = cv2.addWeighted(frame, 1 - MASK_ALPHA, self.color, MASK_ALPHA, 0)
            out[self.fill] = tinted[self.fill]
            out[self.edge] = self.edge_color[self.edge]
        return out


def compose(frame, result, show_masks, boxes=(), drag=None):
    """The video frame with the newest masks and the exemplar boxes on it."""
    height, width = frame.shape[:2]
    out = frame
    if show_masks and result is not None and len(result.masks):
        if result.layer is None or result.layer.size != (width, height):
            result.layer = Layer(result.masks, width, height)
        out = result.layer.draw(frame)
    if out is frame:
        out = frame.copy()
    thick = max(2, round(width / 700))
    for x1, y1, x2, y2, label in boxes:
        cv2.rectangle(out, (round(x1 * width), round(y1 * height)),
                      (round(x2 * width), round(y2 * height)),
                      POSITIVE if label else NEGATIVE, thick)
    if drag is not None:
        (x1, y1), (x2, y2), label = drag
        cv2.rectangle(out, (x1, y1), (x2, y2), POSITIVE if label else NEGATIVE, 1)
    return out


def hud(width, lines):
    """The strip under the video. Kept off the image so it never hides a cell."""
    strip = np.full((HUD_HEIGHT, width, 3), 28, dtype=np.uint8)
    scale = max(0.42, min(0.72, width / 1900 * 0.72))
    for row, (text, color) in enumerate(lines):
        cv2.putText(strip, text, (12, 24 + row * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)
    return strip


def placeholder(args, text):
    frame = np.full((args.height // 2, args.width // 2, 3), 18, dtype=np.uint8)
    cv2.putText(frame, text, (30, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (170, 170, 170), 2, cv2.LINE_AA)
    return frame


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def save_snapshot(args, result):
    """The segmented frame, its masks at the frame's own size, and how they were
    asked for. The .npz is remote_sam3's packed format (maskio.load_masks)."""
    folder = Path(args.out) / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    name = stamp()
    height, width = result.frame.shape[:2]
    if len(result.masks):
        full = np.stack([cv2.resize(m.astype(np.uint8), (width, height),
                                    interpolation=cv2.INTER_NEAREST) > 0
                         for m in result.masks])
    else:
        full = np.zeros((0, height, width), dtype=bool)
    cv2.imwrite(str(folder / f"{name}.png"), result.frame)
    np.savez_compressed(
        folder / f"{name}_masks.npz",
        packed=np.packbits(full, axis=None),
        shape=np.asarray(full.shape, dtype=np.int32),
        scores=np.asarray(result.scores, dtype=np.float32))
    (folder / f"{name}.json").write_text(json.dumps({
        "prompt": result.prompt,
        "threshold": result.threshold,
        "scores": result.scores,
        "segmented_at": list(result.masks.shape[1:]),
        "saved_at": [height, width],
        "server": result.meta,
    }, indent=2))
    return f"saved {folder / name}.png + _masks.npz ({len(full)} masks)"


# -- The window ----------------------------------------------------------------

class Window:

    def __init__(self, args, camera, sender, state):
        self.args, self.camera, self.sender, self.state = args, camera, sender, state
        self.show_masks = True
        self.editing = None             # the text being typed, or None
        self.message, self.message_until = "", 0
        self.drag = None                # ((x1, y1), (x2, y2), label) in frame pixels
        self.frame_size = (args.width, args.height)

    def say(self, text):
        self.message, self.message_until = text, time.monotonic() + 4

    def on_mouse(self, event, x, y, flags, _):
        width, height = self.frame_size
        x, y = min(max(x, 0), width - 1), min(max(y, 0), height - 1)
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            negative = event == cv2.EVENT_RBUTTONDOWN or flags & cv2.EVENT_FLAG_SHIFTKEY
            self.drag = ((x, y), (x, y), 0 if negative else 1)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            self.drag = (self.drag[0], (x, y), self.drag[2])
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP) and self.drag is not None:
            (x1, y1), _, label = self.drag
            self.drag = None
            if abs(x - x1) >= MIN_BOX and abs(y - y1) >= MIN_BOX:
                box = (min(x1, x) / width, min(y1, y) / height,
                       max(x1, x) / width, max(y1, y) / height, label)
                with self.state.lock:
                    self.state.boxes.append(box)

    def on_key(self, key):
        """Returns False to quit."""
        state = self.state
        if self.editing is not None:
            if key in (13, 10):                     # Enter
                with state.lock:
                    state.prompt = self.editing.strip()
                self.editing = None
            elif key == 27:                         # Esc
                self.editing = None
            elif key in (8, 127):                   # Backspace
                self.editing = self.editing[:-1]
            elif 32 <= key < 127:
                self.editing += chr(key)
            return True

        char = chr(key) if 32 <= key < 127 else ""
        if char == "q" or key == 27:
            return False
        if char == "p" or key == 9:
            self.editing = state.prompt
        elif char.isdigit() and char != "0":
            index = int(char) - 1
            if index < len(self.args.presets):
                with state.lock:
                    state.prompt = self.args.presets[index]
        elif char in "[]":
            with state.lock:
                step = -0.05 if char == "[" else 0.05
                state.threshold = min(0.95, max(0.05, state.threshold + step))
        elif char == "c":
            with state.lock:
                state.boxes.clear()
        elif char == " ":
            if state.frozen is None:
                seq, frame = self.camera.latest()
                if frame is not None:
                    state.frozen = (seq, frame)
            else:
                state.frozen = None
        elif char == "m":
            self.show_masks = not self.show_masks
        elif char == "r":
            self.say(self.camera.toggle_recording())
        elif char == "s":
            result = self.sender.latest()
            self.say(save_snapshot(self.args, result) if result else "no result to save yet")
        elif char == "n":
            if self.args.source:
                self.say("--source is set: there is only one source")
            else:
                self.camera.next_device()
                self.say(f"switching to camera {self.camera.device}")
        return True

    def render(self):
        state = self.state
        _, frame = state.frozen or self.camera.latest()
        if frame is None:
            frame = placeholder(self.args, self.camera.status)
        self.frame_size = (frame.shape[1], frame.shape[0])
        result = self.sender.latest()
        prompt, threshold, _, boxes = state.params()
        picture = compose(frame, result, self.show_masks, boxes, self.drag)

        white, dim, warn = (235, 235, 235), (150, 150, 150), (80, 200, 255)
        if self.editing is not None:
            cursor = "_" if int(time.monotonic() * 2) % 2 else " "
            first = (f"prompt> {self.editing}{cursor}   (Enter apply, Esc cancel)", warn)
        else:
            first = (f"prompt: {prompt or '(none - press p)'}", white)
        if result is not None and (prompt or boxes):
            timing = (f"{len(result.scores)} masks   {self.sender.rate:.1f} seg/s   "
                      f"round trip {result.round_trip_ms:.0f} ms "
                      f"(gpu {result.meta.get('infer_ms', 0):.0f})")
        else:
            timing = "no masks"
        flags = "".join([
            "   FROZEN" if state.frozen else "",
            "   REC" if self.camera.recording_to else "",
            "   masks hidden" if not self.show_masks else "",
            f"   {len(boxes)} box" + ("es" if len(boxes) != 1 else "") if boxes else "",
        ])
        second = (f"score >= {threshold:.2f}   {timing}{flags}", white)
        if time.monotonic() < self.message_until:
            third = (self.message, warn)
        else:
            third = (f"{self.camera.status}   |   server: {self.sender.status}", dim)
        return np.vstack([picture, hud(picture.shape[1], [first, second, third])])

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, 1280, 720 + HUD_HEIGHT * 1280 // self.args.width)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        deadline = time.monotonic() + self.args.duration if self.args.duration else None
        while deadline is None or time.monotonic() < deadline:
            cv2.imshow(WINDOW, self.render())
            key = cv2.waitKey(15)
            if key != -1 and not self.on_key(key & 0xFF):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


def run_headless(args, camera, sender, state):
    """No window: wait for N results, print them, save the last composite."""
    seen, last = 0, None
    deadline = time.monotonic() + args.timeout + 10
    while seen < args.headless and time.monotonic() < deadline:
        if sender.results > seen and sender.latest() is not last:
            last = sender.latest()
            seen = sender.results
            print(f"result {seen}: {len(last.scores)} masks {last.masks.shape} "
                  f"scores {[round(s, 2) for s in last.scores[:5]]} "
                  f"round trip {last.round_trip_ms:.0f} ms  meta {last.meta.get('infer_ms')}")
            deadline = time.monotonic() + args.timeout + 10
        time.sleep(0.01)
    if last is None:
        raise SystemExit(f"no result: camera '{camera.status}', server '{sender.status}'")
    _, _, _, boxes = state.params()
    path = Path(args.out) / "headless.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), compose(last.frame, last, True, boxes))
    print(f"composite : {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_argument_group("video source")
    source.add_argument("--device", type=int, default=0, help="camera index (see --list)")
    source.add_argument("--source", help="a video or image file instead of a camera")
    source.add_argument("--width", type=int, default=1920)
    source.add_argument("--height", type=int, default=1080)
    source.add_argument("--fps", type=int, default=30)
    source.add_argument("--list", action="store_true",
                        help="open each camera in turn, save a picture from each, and exit")
    source.add_argument("--max-devices", type=int, default=4)

    server = parser.add_argument_group("server")
    server.add_argument("--host", default="127.0.0.1", help="the Mac end of the ssh tunnel")
    server.add_argument("--port", type=int, default=DEFAULT_PORT)
    server.add_argument("--token", default=os.environ.get("SAM_LISTENER_TOKEN"),
                        help="the token the server printed (or $SAM_LISTENER_TOKEN)")
    server.add_argument("--timeout", type=float, default=30, help="seconds to wait for one frame")
    server.add_argument("--send-size", type=int, default=1008,
                        help="longer side of the frame sent, in pixels; SAM3 works at 1008")
    server.add_argument("--jpeg-quality", type=int, default=90)

    prompt = parser.add_argument_group("prompt")
    prompt.add_argument("--prompt", default="", help="the prompt to start with")
    prompt.add_argument("--prompts", default="",
                        help="comma-separated presets for keys 1-9, e.g. 'cell wall,nucleus,chloroplast'")
    prompt.add_argument("--threshold", type=float, default=0.5,
                        help="keep detections at or above this score")
    prompt.add_argument("--mask-threshold", type=float, default=0.5,
                        help="binarisation cutoff for each mask's pixels")

    parser.add_argument("--out", default=str(Path(__file__).resolve().parent),
                        help="where recordings/ and snapshots/ go")
    parser.add_argument("--headless", type=int, default=0, metavar="N",
                        help="no window: take N results, save a composite, exit")
    parser.add_argument("--duration", type=float, default=0,
                        help="close the window after this many seconds")
    args = parser.parse_args()
    args.presets = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not args.prompt and args.presets:
        args.prompt = args.presets[0]
    return args


def main():
    args = parse_args()
    if args.list:
        list_devices(args)
        return
    state = State(args)
    camera = Camera(args)
    sender = Sender(args, camera, state)
    sender.gpu = "ok"
    camera.start()
    sender.start()
    try:
        if args.headless:
            run_headless(args, camera, sender, state)
        else:
            Window(args, camera, sender, state).run()
    finally:
        camera.stop.set()
        camera.reopen.set()
        sender.stop.set()
        camera.join(timeout=3)


if __name__ == "__main__":
    main()
