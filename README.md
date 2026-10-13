# SamListener — live SAM3 over a capture card

Point a microscope camera at a USB capture card, and see SAM3's masks drawn
over the live video, for a text prompt you can retype while it runs.

The Mac only captures and draws. SAM3 runs on a HiPerGator GPU; frames travel
there and masks come back through an ssh tunnel.

```
microscope ─HDMI─▶ capture card ─USB─▶ listen.py (Mac)  ══ssh tunnel══▶  server.py (GPU node)
                                        window + prompt   ◀══ masks ════  SAM3 in memory
```

```
listen.py              the Mac: video window, prompt, boxes, recording, snapshots
server.py              the GPU node: SAM3 behind one HTTP endpoint
protocol.py            the wire format both ends share
slurm/listener.sbatch  runs server.py on one GPU
```

## 1. On HiPerGator

The server needs nothing installed. It borrows the conda env and the SAM3
weights that `remote_sam3`'s `setup_env.sh` already put in the `flower_sam3`
checkout, and expects to sit next to it:

```
~/blue_guralnick/neal.nevyn/
├── flower_sam3/     .conda/sam3 and hf_cache live here
└── SamListener/     this repo
```

Copy the repo up from the Mac (the repo is private, so a plain `git clone` on
HiPerGator would ask for credentials):

```bash
rsync -av --exclude .venv --exclude .git --exclude recordings --exclude snapshots \
    ~/Documents/code/repos/SamListener/ \
    hipergator:blue_guralnick/neal.nevyn/SamListener/
```

Then, on HiPerGator:

```bash
cd ~/blue_guralnick/neal.nevyn/SamListener
sbatch slurm/listener.sbatch
tail -f logs/sam_listener-*.out
```

When the model has loaded, the log prints the two commands for the Mac, with
this job's node name and a fresh token filled in:

```
listening  : c0907a-s23:8766

On the Mac, in two terminals:

    ssh -N -L 8766:c0907a-s23:8766 hipergator
    .venv/bin/python listen.py --port 8766 --token 3qX...
```

`scancel <jobid>` when you are done. The job otherwise runs for its four hours.

| to change | how |
|---|---|
| session length | `sbatch --time=08:00:00 slurm/listener.sbatch` |
| GPU | `sbatch --partition=hpg-b200 slurm/listener.sbatch` |
| speed over exact edges | `BF16=1 sbatch slurm/listener.sbatch` |
| where `flower_sam3` is | `SAM3_PROJECT=/path/to/flower_sam3 sbatch ...` |
| port | `PORT=8770 sbatch ...` |

## 2. On the Mac

One-time setup (already done on this machine):

```bash
cd ~/Documents/code/repos/SamListener
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Find which camera index is the capture card. This opens each camera for a
moment, so the webcam light comes on and a nearby iPhone wakes up:

```bash
.venv/bin/python listen.py --list        # pictures land in devices/
```

The first time, macOS asks whether the terminal may use the camera. Allow it
and run the command again.

Then the two commands from the server's log, adding `--device N` if the card
is not camera 0, and any presets you want on the number keys:

```bash
ssh -N -L 8766:<node>:8766 hipergator
.venv/bin/python listen.py --token <token> --device 0 \
    --prompts "cell,nucleus,chloroplast,stoma,cell wall"
```

The viewer runs without the server too: it shows the video and says
`no server on 127.0.0.1:8766` until the tunnel is up, then starts by itself.

### Keys

| key | does |
|---|---|
| `p` or Tab | edit the prompt — type, Enter to apply, Esc to cancel |
| `1`…`9` | jump to a preset from `--prompts` |
| `[` `]` | score threshold down / up by 0.05 |
| drag | box an example: "segment things like this" |
| shift-drag, right-drag | box a counter-example: "not this" |
| `c` | clear the boxes |
| space | freeze the frame, so you can box something on a moving specimen |
| `m` | masks on / off |
| `r` | start / stop recording the raw video (no overlay) to `recordings/` |
| `s` | snapshot the segmented frame, its masks and the prompt to `snapshots/` |
| `n` | next camera |
| `q` or Esc | quit |

### What gets saved

Nothing, unless you ask.

- `r` writes `recordings/<time>.mp4`: the frames exactly as captured, without
  the overlay, so they can be segmented again later with a better prompt.
- `s` writes three files to `snapshots/`: `<time>.png` (the frame),
  `<time>_masks.npz` (its masks, at the frame's size) and `<time>.json` (the
  prompt, threshold and scores). The `.npz` is `remote_sam3`'s packed format,
  so `maskio.load_masks()` reads it.

## How it behaves

- **Only the newest frame is sent.** While the GPU is busy, frames are shown
  but never queued. The masks trail the video by one round trip and no more;
  the status line shows that round trip and the GPU's share of it.
- **The prompt travels with each frame**, so a new prompt takes effect on the
  very next frame. The server keeps nothing between frames.
- **Boxes are tied to the picture, not the object.** A box says "the thing at
  this spot in the frame". Move the stage and it points at something else —
  freeze, box, look, then clear.
- **Frames are shrunk to 1008 px on the long side** before sending
  (`--send-size`), which is the resolution SAM3 works at. The masks are scaled
  back up over the full-resolution video.

## Testing without a GPU or a camera

`server.py --mock` loads no model and returns the dark and the bright regions
of each frame as fake detections:

```bash
SAM_LISTENER_TOKEN=test .venv/bin/python server.py --mock --host 127.0.0.1 &
.venv/bin/python listen.py --token test --source some_clip.mp4 --prompt anything
```

`--source` takes a video (looped) or a still image. `--headless N` skips the
window, waits for N results and writes `headless.png`.

## Security

The server listens on the compute node's network so the login node can reach
it, which means other cluster users can reach it too. Every request must carry
the token printed in the job's log; a new one is generated for each job. The
traffic between the Mac and HiPerGator is inside ssh. Between the login node
and the compute node it is plain HTTP on the cluster's internal network.
