# Shard Frontend

The frontend renders synthesized generations from the Python backend as an
interactive Three.js scene. It connects to the backend over
`ws://localhost:8765`.

## Run locally

From this directory:

```powershell
npm install
npm start
```

In another terminal, start the backend from the repository root:

```powershell
python Backend/src/main.py
```

The Webpack development server listens on `http://localhost:9000`. To create a
production bundle, run `npm run build`; generated files are written to `dist/`.
The generated bundle uses relative, runtime-resolved asset URLs so it can be
served from the repository root or a project subpath.

To inspect bundle composition, run `npm run analyze`.

## Source layout

- `src/main.js` initializes the scene and wires the application components.
- `src/components/Navigation.js` owns the hamburger menu and its hover messages.
- `src/components/KeyboardControls.js` handles keyboard input and movement keys.
- `src/components/ModeTutorialSystem.js` manages first-use help messages.
- `src/components/TitleScreen.js` renders the animated launch screen.
- `src/audio/audio-export.js` encodes and mixes exported library audio.
- `src/styles.css` imports the ordered style sections in `src/styles/`; keep
  their import order stable because the cascade is shared across the UI.
