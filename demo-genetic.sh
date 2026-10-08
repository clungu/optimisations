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

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from optimisations.animations import animate
from optimisations.figures import Figure
from optimisations.functions import himmelblau
from optimisations.genetic import genetic_algo, get_params
from optimisations.optimizers import optimize
from optimisations.renderers import decorate_with_derivative_free_plot

frame_count = 80
interval = 100
function = himmelblau()


def render_population(name, history, figure):
    decorate_with_derivative_free_plot(name, history, figure)
    population = history[-1]
    best_loss = min(function(x, y) for x, y in population)
    figure.ax_2d.set_title(f"Generation {len(history) - 1} | Best f = {best_loss:.6g}")


run = (
    optimize(function)
    .using(
        genetic_algo(
            population_size=50, max_offset=4, encoding="ieee754",
            operators="guarded", mutation_chance=0.48 / 128, seed=7,
        ),
        name="Guarded float64 GA",
        derivatives_based=False,
        render_decorator=render_population,
    )
    .start_from([-1.0, 1.0])
)
initial_best = min(function(x, y) for x, y in get_params(run.state))

# Keep all PNG frames embedded rather than dropping frames at Matplotlib's limit.
with matplotlib.rc_context({"animation.embed_limit": 100}):
    video = animate(
        run, frames=frame_count, interval=interval, output="js", show_diagnostics=True,
        figure=Figure(fig=plt.figure(figsize=(14, 6)), contour_log_scale=False, angle=-45),
    )

population = get_params(run.state)
best = population[np.argmax(run.state.fitness)]
best_loss = float(function(*best))
summary = (
    f"80 generations, 50 individuals, seed 7. Guarded IEEE-754 genomes, "
    f"0.48 expected bit flips per offspring. Search bounds: x in [-5, 3], y in [-3, 5]. "
    f"Best loss: {initial_best:.8g} to {best_loss:.8g}; "
    f"best point: ({best[0]:.8g}, {best[1]:.8g})."
)

encoded_frames = re.findall(r'frames\[\d+\] = "data:image/png;base64,([^"]+)"', video)
if len(encoded_frames) != frame_count:
    raise ValueError(f"Expected {frame_count} animation frames, found {len(encoded_frames)}")
if len(run.history) != frame_count + 1:
    raise ValueError(f"Expected {frame_count + 1} history states, found {len(run.history)}")

output = Path("output")
output.mkdir(exist_ok=True)
html_path = output / "himmelblau-genetic.html"
html_path.write_text(
    '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
    '<title>Himmelblau - guarded float64 genetic algorithm</title></head><body>\n'
    '<h1>Himmelblau: guarded float64 genetic algorithm</h1>\n'
    f"<p>{summary}</p>\n"
    + video + "\n</body></html>\n",
    encoding="utf-8",
)

frames = []
for encoded_frame in encoded_frames:
    with Image.open(BytesIO(base64.b64decode(encoded_frame.replace("\\\n", "")))) as image:
        width = 1120
        height = round(image.height * width / image.width)
        frames.append(
            image.convert("RGB")
                 .resize((width, height), Image.Resampling.LANCZOS)
                 .quantize(colors=128)
        )

gif_path = output / "himmelblau-genetic.gif"
frames[0].save(
    gif_path, save_all=True, append_images=frames[1:],
    duration=interval, loop=0, optimize=True, disposal=1,
)
print(f"\n{summary}")
for path in (html_path, gif_path):
    print(f"Saved {path} ({path.stat().st_size:,} bytes)")
PY
