# Alzheimer's MRI Classifier — Full Stack App

Two parts that run together:
- `backend/`  — Python + FastAPI. Loads your trained `quantum_gat_sage_final.pth`
  checkpoint and serves predictions.
- `frontend/` — React (Vite). The upload/results website in your browser.

This has already been tested end-to-end against your real checkpoint file
(the .pth you uploaded is already sitting in `backend/checkpoints/`).

## One-time setup

### 1. Backend

```
cd backend
python -m venv venv
```
Turn on the virtual environment (do this every time before running the backend):
- Windows:  `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

Then install dependencies (this step takes a few minutes, only needed once):
```
pip install -r requirements.txt
```

### 2. Frontend

In a separate terminal:
```
cd frontend
npm install
```

## Running the app (every time)

You need **two terminals open at the same time**.

**Terminal 1 — backend:**
```
cd backend
venv\Scripts\activate        (Mac/Linux: source venv/bin/activate)
uvicorn main:app --reload --port 8000
```
Wait until you see it says it's running on port 8000. Leave this terminal open.

**Terminal 2 — frontend:**
```
cd frontend
npm run dev
```
It will print a link, usually `http://localhost:5173`. Open that in your browser.

## Using it

1. Click the upload box, choose an MRI slice (jpg/png) — you can use any image
   from your `Alzheimer_s Dataset` folders (NonDemented / MildDemented /
   ModerateDemented / VeryMildDemented) to try it.
2. Click "Analyze MRI".
3. Wait a few seconds (the quantum circuit runs on CPU, so it's not instant).
4. You'll see: predicted stage, confidence, a probability bar chart across
   all 4 classes, the skull-stripped preview, and a patch-attention heatmap.

## Quick check that the backend works on its own

Before trying the frontend, you can sanity-check the backend by itself:
```
http://127.0.0.1:8000/docs
```
This opens an auto-generated test page. Try the `/predict` endpoint there with
a sample image before worrying about the frontend at all.

## If something goes wrong

| Symptom | Fix |
|---|---|
| Frontend shows "Could not reach the analysis server" | Terminal 1 (backend) isn't running — check it for errors |
| Backend fails to start with a `state_dict` error | You replaced the checkpoint file with a different one; the architecture in `model.py` must match it exactly |
| `pip install -r requirements.txt` fails on `torch-geometric` | Run `pip install torch==2.3.0` by itself first, then `pip install -r requirements.txt` again |
| Everything works but takes 3–10 seconds per image | Normal — PennyLane's quantum simulator runs on CPU regardless of your hardware |
| `ModuleNotFoundError: No module named 'torch'` | You forgot to activate the virtual environment (`venv\Scripts\activate`) before installing/running |
