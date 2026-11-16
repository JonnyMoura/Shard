# Shard

Shard combines an evolutionary audio-synthesis backend with a 3D sound-particle visualization frontend.

## Backend

The Python backend is in [`Backend/`](Backend/). Install its dependencies and run the evolutionary process:

```powershell
cd Backend
python -m pip install -r requirements.txt
python src/main.py
```

## Frontend

The JavaScript frontend is in [`Frontend/`](Frontend/). Install dependencies and start the development server:

```powershell
cd Frontend
npm install
npm start
```

See each component's README for more details.
