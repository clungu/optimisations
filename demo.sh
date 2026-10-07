#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -x .venv/bin/python ]]; then
    python=.venv/bin/python
else
    python=python3
fi

MPLBACKEND=Agg "$python" - <<'PY'
import base64
from io import BytesIO
from pathlib import Path
import re

from jax.example_libraries.optimizers import sgd
from PIL import Image

from optimisations.functions import himmelblau
from optimisations.optimizers import optimize
from optimisations.animations import animate

video = animate(
    optimize(himmelblau())
        .using(sgd(step_size=0.01))
        .start_from([0.0003, 0.01]),
    frames=20,
    output='js'
)

path = Path('output/README-example.html')
path.parent.mkdir(exist_ok=True)
path.write_text(
    '<!doctype html>\n'
    '<html lang="en"><head><meta charset="utf-8">'
    '<title>Optimisations README example</title></head><body>\n'
    + video + '\n</body></html>\n',
    encoding='utf-8',
)
print(f'\nSaved {path} ({path.stat().st_size:,} bytes)')

encoded_frames = re.findall(
    r'frames\[\d+\] = "data:image/png;base64,([^"]+)"', video
)
if len(encoded_frames) != 20:
    raise ValueError(f'Expected 20 animation frames, found {len(encoded_frames)}')

frames = []
for encoded_frame in encoded_frames:
    with Image.open(BytesIO(base64.b64decode(encoded_frame.replace('\\\n', '')))) as image:
        width = 900
        height = round(image.height * width / image.width)
        frames.append(
            image.convert('RGB')
                 .resize((width, height), Image.Resampling.LANCZOS)
                 .quantize(colors=128)
        )

preview = Path('README-example.gif')
frames[0].save(
    preview, save_all=True, append_images=frames[1:],
    duration=50, loop=0, optimize=True, disposal=1,
)
print(f'Saved {preview} ({preview.stat().st_size:,} bytes)')
PY
