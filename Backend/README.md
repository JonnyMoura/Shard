# Backend

The Python backend generates audio with an evolutionary MAP-Elites algorithm and serves generations to the frontend over WebSocket.

From the repository root, install the dependencies and start the backend:

```powershell
python -m pip install -r Backend/requirements.txt
python Backend/src/main.py
```

The frontend connects to `ws://localhost:8765`. See the [project README](../README.md) for frontend setup.

Run an evaluation from this directory with:

```powershell
python -m src.evaluation_runner --runs 1 --generations 5 --pop 64 --sr 44100 --dims 20 20 20
```

The population, sample rate, and three archive dimensions control the actual
evaluation run. Relative catalog, audio, and output paths resolve from
`Backend/`.

## Source layout

- `src/main.py` coordinates the live application and its WebSocket/evolution
  loop.
- `src/generation_payload.py` builds the generation message sent to the
  frontend.
- `src/synthesis_pipeline.py` resolves source paths and runs synthesis jobs.
- `src/fitness.py` calculates descriptor-based automatic fitness;
  `src/evolutionary_module.py` configures the archive and emitters.
- `src/feedback_validation.py` validates that scores match the generation
  awaiting evaluation.
- `tests/` contains focused regression tests. Run them from `Backend/` with:

  ```powershell
  python -m unittest discover -s tests
  ```