import os, io, json, base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from shapely import wkt
from sklearn.cluster import DBSCAN
from torchvision import transforms
from torchvision.models import swin_t
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
LABELS_DIR = ROOT / "labels"
OUTPUT_DIR.mkdir(exist_ok=True)
LABELS_DIR.mkdir(exist_ok=True)
load_dotenv(ROOT / "backend" / ".env")

MODEL_PATH = os.getenv("MODEL_PATH", r"C:\Users\Indhu\Downloads\disaster_dashboard_project\swin_transformer_damage.pth")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
CLASS_NAMES = ["destroyed", "major_damage", "minor_damage", "no_damage"]
DISPLAY_NAMES = {
    "destroyed": "Destroyed", "major_damage": "Major Damage",
    "minor_damage": "Minor Damage", "no_damage": "No Damage"
}
COLORS_BGR = {
    "destroyed": (0, 0, 255), "major_damage": (0, 165, 255),
    "minor_damage": (0, 255, 255), "no_damage": (0, 200, 80)
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
MODEL_ERROR = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = FastAPI(title="Disaster Response AI Dashboard")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"]
)

def load_model():
    global MODEL, MODEL_ERROR
    if MODEL is not None:
        return MODEL
    if not Path(MODEL_PATH).is_file():
        MODEL_ERROR = f"Model checkpoint not found: {MODEL_PATH}. Set MODEL_PATH in backend/.env."
        raise RuntimeError(MODEL_ERROR)
    model = swin_t(weights=None, num_classes=4)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # Support either a raw state_dict or a checkpoint wrapper.
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported checkpoint format: expected a state_dict dictionary.")
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in checkpoint.items()}
    model.load_state_dict(cleaned, strict=True)
    model.to(DEVICE).eval()
    MODEL = model
    return MODEL

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "device": str(DEVICE),
        "model_path": MODEL_PATH,
        "model_file_exists": Path(MODEL_PATH).is_file(),
        "groq_configured": bool(GROQ_API_KEY),
        "note": "Predictions require the matching Swin-T checkpoint and xView2 building polygon JSON."
    }

def decode_image(raw: bytes):
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read the uploaded image. Please upload a valid PNG/JPG.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def find_matching_annotation(image_name: str):
    if not image_name:
        return None
    stem = Path(image_name).stem.lower()
    candidates = []
    if LABELS_DIR.exists():
        for p in sorted(LABELS_DIR.rglob("*.json")):
            name = p.stem.lower()
            if name == stem:
                return p
            if name in {stem + "_annotations", stem + "_annotation"}:
                candidates.append(p)
    return candidates[0] if candidates else None

def run_assessment(rgb_image, annotation_data, image_name):
    model = load_model()
    features = annotation_data.get("features", {}).get("xy", [])
    if not features:
        raise ValueError("Annotation JSON must contain features.xy building polygons, as in xView2.")
    h_img, w_img = rgb_image.shape[:2]
    rows = []
    for feature in features:
        props = feature.get("properties", {})
        if props.get("feature_type") != "building":
            continue
        try:
            poly = wkt.loads(feature["wkt"])
            if poly.is_empty or not hasattr(poly, "exterior"):
                continue
            coords = np.asarray(list(poly.exterior.coords), dtype=np.int32)
            x, y, w, h = cv2.boundingRect(coords)
            if w <= 0 or h <= 0:
                continue
            cx, cy = float(poly.centroid.x), float(poly.centroid.y)
            # xView2 annotation coordinates should align with the uploaded image.
            x1, y1 = max(0, x - 50), max(0, y - 50)
            x2, y2 = min(w_img, x + w + 50), min(h_img, y + h + 50)
            crop = rgb_image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            tensor = transform(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                probs = torch.softmax(model(tensor), dim=1)[0]
            idx = int(torch.argmax(probs).item())
            rows.append({
                "building_id": props.get("uid", f"B{len(rows)+1:05d}"),
                "center_x": cx, "center_y": cy, "x": x, "y": y, "width": w, "height": h,
                "predicted_damage": CLASS_NAMES[idx], "confidence": round(float(probs[idx].item()), 4),
                "polygon_wkt": feature["wkt"]
            })
        except Exception:
            continue
    if not rows:
        raise ValueError("No valid building polygons were processed. Check that the JSON and image correspond.")
    coords = np.asarray([[r["center_x"], r["center_y"]] for r in rows], dtype=float)
    labels = DBSCAN(eps=80, min_samples=3).fit_predict(coords)
    for r, label in zip(rows, labels):
        r["cluster_id"] = int(label)

    counts = {name: sum(r["predicted_damage"] == name for r in rows) for name in CLASS_NAMES}
    total = len(rows)
    affected = total - counts["no_damage"]
    affected_pct = round(100 * affected / total, 2)
    cluster_ids = sorted(set(int(x) for x in labels if x != -1))
    cluster_results = []
    for cid in cluster_ids:
        group = [r for r in rows if r["cluster_id"] == cid]
        c = {name: sum(r["predicted_damage"] == name for r in group) for name in CLASS_NAMES}
        n = len(group)
        affected_c = n - c["no_damage"]
        severity = ((c["destroyed"]*3 + c["major_damage"]*2 + c["minor_damage"]) / (n*3))*100
        damaged_pct = affected_c / n * 100
        destroyed_pct = c["destroyed"] / n * 100
        score = .5*severity + .3*damaged_pct + .2*destroyed_pct
        priority = "CRITICAL" if score >= 70 else "HIGH" if score >= 45 else "MEDIUM" if score >= 20 else "LOW"
        cluster_results.append({
            "cluster_id": cid, "total_buildings": n, "destroyed": c["destroyed"],
            "major_damage": c["major_damage"], "minor_damage": c["minor_damage"],
            "no_damage": c["no_damage"], "affected_buildings": affected_c,
            "damage_percentage": round(damaged_pct, 2), "severity_score": round(severity, 2),
            "priority_score": round(score, 2), "priority": priority,
            "cluster_x": round(float(np.mean([r["center_x"] for r in group])), 2),
            "cluster_y": round(float(np.mean([r["center_y"] for r in group])), 2)
        })
    cluster_results.sort(key=lambda x: x["priority_score"], reverse=True)
    priority_by_id = {c["cluster_id"]: c["priority"] for c in cluster_results}
    for r in rows:
        r["priority"] = priority_by_id.get(r["cluster_id"], "ISOLATED")

    # Render actual polygon overlay on the uploaded image.
    overlay = cv2.cvtColor(rgb_image.copy(), cv2.COLOR_RGB2BGR)
    for r in rows:
        try:
            poly = wkt.loads(r["polygon_wkt"])
            points = np.asarray(list(poly.exterior.coords), dtype=np.int32)
            color = COLORS_BGR[r["predicted_damage"]]
            cv2.polylines(overlay, [points], True, color, 2)
            cv2.circle(overlay, (int(r["center_x"]), int(r["center_y"])), 3, color, -1)
        except Exception:
            pass
    overlay_path = OUTPUT_DIR / "damage_map.png"
    cv2.imwrite(str(overlay_path), overlay)
    cluster_points = [{"x": r["center_x"], "y": r["center_y"], "cluster_id": r["cluster_id"],
                       "damage": r["predicted_damage"], "priority": r["priority"]} for r in rows]
    resources = {
        "ambulances": max(1, int(np.ceil((counts["destroyed"] + .5*counts["major_damage"])/3))),
        "rescue_teams": max(1, int(np.ceil((counts["destroyed"]*1.5 + counts["major_damage"])/3))),
        "fire_rescue_trucks": max(1, int(np.ceil((counts["destroyed"]+counts["major_damage"])/4))),
        "medical_teams": max(1, int(np.ceil(affected/15))),
        "relief_camps": max(1, int(np.ceil(affected/30)))
    }
    summary = {
        "image": image_name, "total_buildings": total, "destroyed": counts["destroyed"],
        "major_damage": counts["major_damage"], "minor_damage": counts["minor_damage"],
        "no_damage": counts["no_damage"], "affected_buildings": affected,
        "affected_percentage": affected_pct, "total_clusters": len(cluster_ids),
        "isolated_buildings": int(sum(r["cluster_id"] == -1 for r in rows)),
        "overall_severity": "CRITICAL" if counts["destroyed"] >= 15 or affected_pct >= 50 else "HIGH" if counts["destroyed"] >= 8 or affected_pct >= 30 else "MEDIUM" if affected_pct >= 15 else "LOW",
        "rescue_difficulty": "VERY HARD" if affected_pct >= 50 else "HARD" if affected_pct >= 30 else "MODERATE" if affected_pct >= 15 else "RELATIVELY EASY",
        "estimated_resources": resources, "clusters": cluster_results,
        "most_important_cluster": cluster_results[0] if cluster_results else None,
        "cluster_points": cluster_points,
        "damage_map_base64": base64.b64encode(overlay_path.read_bytes()).decode("ascii")
    }
    (OUTPUT_DIR / "disaster_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # CSV exports are useful for the report/demo.
    import csv
    with (OUTPUT_DIR / "building_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    return summary

def generate_llama_report(summary):
    if not GROQ_API_KEY:
        cluster_text = "No cluster priorities were generated by Groq because no API key is configured."
        if summary.get("clusters"):
            top_cluster = summary["clusters"][0]
            cluster_text = (
                f"Highest-priority cluster: Cluster {top_cluster['cluster_id']} with {top_cluster['total_buildings']} buildings, "
                f"severity {top_cluster['severity_score']:.1f}%, priority {top_cluster['priority']}"
            )
        return (
            "Groq Llama report: not enabled (GROQ_API_KEY is empty).\n\n"
            "Damage analysis completed successfully using the uploaded image and matching labels.\n\n"
            f"Overview: {summary.get('total_buildings', 0)} buildings assessed, "
            f"{summary.get('affected_buildings', 0)} affected ({summary.get('affected_percentage', 0)}%).\n"
            f"Damage counts: destroyed={summary.get('destroyed', 0)}, major_damage={summary.get('major_damage', 0)}, "
            f"minor_damage={summary.get('minor_damage', 0)}, no_damage={summary.get('no_damage', 0)}.\n"
            f"{cluster_text}\n\n"
            "Estimated resources: " + ", ".join(
                f"{k.replace('_',' ').title()}={v}" for k, v in (summary.get('estimated_resources') or {}).items()
            ) + ".\n\n"
            "To enable the Llama report, add a valid GROQ_API_KEY in backend/.env and rerun the assessment."
        )
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        system = ("You are a disaster-response reporting assistant. Use only supplied assessment data; never "
             "invent or change numbers or incident details. Write a polished, professional report in natural, "
             "complete sentences with numbered section headings and no report title. Use plain text without "
             "Markdown styling, tables, generic disclaimers, or placeholder text.")
        user = ("Write these sections in order: 1. Overview, 2. Damage Summary, 3. Spatial Clusters, 4. "
            "Highest-Priority Cluster, 5. Estimated Resources, and 6. Immediate Actions. Use short paragraphs "
            "for the overview and damage summary. Write Estimated Resources as one natural sentence containing "
            "every resource type and its quantity, not as bullets. Describe each cluster in one readable sentence, "
            "not as a table or field list. Keep Immediate Actions as a short numbered list. Preserve all "
            "meaningful assessment counts, percentages, severity, rescue difficulty, and isolated-building "
            "counts. In the overview, state the disaster type based on the image filename only when it is clear "
            "(for example, 'socal-fire' means wildfire); do not include an incident description or repeat the "
            "image reference. If unclear, say the disaster type is not specified. In the highest-priority "
            "section, write one sentence naming only the highest-ranked cluster(s) with affected buildings; "
            "include tied IDs and add no rationale or other details. If none have affected buildings, say so. "
            "Keep actions high-level and do not assume survivors, evacuations, active fires, structural safety, "
            "or facility locations unless supplied in the data. Do not add a disclaimer or placeholder. Data:\n") + json.dumps(
            {k:v for k,v in summary.items() if k != "damage_map_base64" and k != "cluster_points"}, indent=2)
        resp = client.chat.completions.create(
            model=GROQ_MODEL, messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.2, max_tokens=1800
        )
        return resp.choices[0].message.content or "The LLM returned an empty report."
    except Exception as e:
        return f"Groq report failed: {e}. Verify the API key and GROQ_MODEL in backend/.env."

@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...), annotations: Optional[UploadFile] = File(None)):
    try:
        image_bytes = await image.read()
        rgb = decode_image(image_bytes)

        if annotations is not None:
            json_bytes = await annotations.read()
            annotation_data = json.loads(json_bytes.decode("utf-8-sig"))
        else:
            annotation_path = find_matching_annotation(image.filename or "")
            if annotation_path is None:
                raise ValueError(
                    f"No matching annotation JSON was found for '{image.filename}' in the project labels folder. "
                    "Use the same base filename in labels/*.json."
                )
            annotation_data = json.loads(annotation_path.read_text(encoding="utf-8-sig"))

        summary = run_assessment(rgb, annotation_data, image.filename or "uploaded_image")
        summary["llm_report"] = generate_llama_report(summary)
        # Save report separately.
        (OUTPUT_DIR / "disaster_response_report.txt").write_text(summary["llm_report"], encoding="utf-8")
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"message": "Disaster dashboard backend is running. Open frontend/index.html in your browser."}
