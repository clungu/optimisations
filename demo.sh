#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -x .venv/bin/python ]]; then
    python=.venv/bin/python
else
    python=python3
fi

MPLBACKEND=Agg "$python" - <<'PY'
from pathlib import Path

from jax.example_libraries.optimizers import sgd

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
PY
