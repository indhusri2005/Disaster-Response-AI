# Disaster Intelligence Dashboard 

## Included
- Dark dashboard UI (`frontend/index.html`)
- FastAPI backend (`backend/main.py`)
- Swin-T inference using your saved checkpoint
- xView2 polygon extraction from the matching annotation JSON
- DBSCAN clustering and priority scoring
- Optional Groq Llama report generation
- Outputs saved under `outputs/`

## 1. Put your model path in the backend config
The default path is:
`C:\Users\Indhu\Downloads\swin_transformer_damage.pth`

If the model is elsewhere, edit `MODEL_PATH` in `backend/.env`.

## 2. Set up Groq (optional)
Open `backend/.env` and set `GROQ_API_KEY=your_new_key`.
Do not paste API keys into source code or commit `.env`. The previously uploaded Python file contained a hard-coded Groq key; revoke/rotate that key in Groq before using this project.

## 3. Run
Double-click `run.bat`. The first run installs Python dependencies and may take several minutes. Keep the terminal open.
Open `frontend/index.html` in Chrome/Edge. If browser file restrictions prevent API calls, use the local frontend server in a second terminal:
`py -m http.server 5500 --directory frontend`
Then visit `http://127.0.0.1:5500`.

Backend health endpoint: `http://127.0.0.1:8000/api/health`
FastAPI docs: `http://127.0.0.1:8000/docs`

## 4. Upload inputs
Upload both:
1. A post-disaster PNG/JPG satellite image.
2. Its matching xView2 JSON annotation file with `features.xy` entries whose `properties.feature_type` is `building` and whose `wkt` polygons align to that image.

The model class order is assumed to be:
0 destroyed, 1 major_damage, 2 minor_damage, 3 no_damage.
This must match the training code used to create your checkpoint.

## Notes
- Strict checkpoint loading is intentional: incompatible checkpoints fail clearly rather than silently giving misleading predictions.
- Inference uses CPU if CUDA is unavailable; processing many building polygons can be slow.
- This dashboard is a research prototype, not a certified emergency-management system. Resource estimates and AI recommendations require human validation.
