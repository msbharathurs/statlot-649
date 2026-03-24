"""
StatLot 649 - FastAPI Backend
Endpoints: ingest, features, predict, backtest, results
"""
import os
import json
import time
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from db.database import get_db, init_db
from db.models import DrawRecord, FeatureRecord, BacktestRun, PredictionRecord
from engine.features import build_feature_row
from engine.backtest import run_backtest
from engine.candidate_gen import generate_candidates, expand_pool_to_combos
from engine.models import MonteCarloCandidateScorer

app = FastAPI(title="StatLot 649 Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# ── Draw Ingestion ─────────────────────────────────────────────────────────────
class DrawInput(BaseModel):
    draw_number: int
    n1: int; n2: int; n3: int; n4: int; n5: int; n6: int
    additional: Optional[int] = None
    draw_date: Optional[str] = None

class BulkDrawInput(BaseModel):
    draws: list[DrawInput]

@app.post("/draws/bulk")
def ingest_draws(payload: BulkDrawInput):
    db = next(get_db())
    inserted, skipped = 0, 0
    for d in payload.draws:
        existing = db.query(DrawRecord).filter_by(draw_number=d.draw_number).first()
        if existing:
            skipped += 1
            continue
        nums = sorted([d.n1,d.n2,d.n3,d.n4,d.n5,d.n6])
        rec = DrawRecord(
            draw_number=d.draw_number,
            draw_date=d.draw_date,
            n1=nums[0],n2=nums[1],n3=nums[2],n4=nums[3],n5=nums[4],n6=nums[5],
            additional=d.additional,
        )
        db.add(rec)
        inserted += 1
    db.commit()
    return {"inserted": inserted, "skipped": skipped, "total": inserted+skipped}

@app.get("/draws")
def get_draws(limit: int = 100, offset: int = 0):
    db = next(get_db())
    rows = db.query(DrawRecord).order_by(DrawRecord.draw_number).offset(offset).limit(limit).all()
    return {"count": len(rows), "draws": [r.to_dict() for r in rows]}

@app.get("/draws/count")
def count_draws():
    db = next(get_db())
    return {"count": db.query(DrawRecord).count()}


# ── Feature Engineering ────────────────────────────────────────────────────────
@app.post("/features/build")
def build_features():
    """Build the full feature lake for all draws."""
    db = next(get_db())
    draws_raw = db.query(DrawRecord).order_by(DrawRecord.draw_number).all()
    draws = [r.to_dict() for r in draws_raw]

    # Delete existing features
    db.query(FeatureRecord).delete()
    db.commit()

    built = 0
    for i in range(len(draws)):
        feat = build_feature_row(i, draws)
        rec = FeatureRecord(draw_number=feat["draw_number"], features_json=json.dumps(feat))
        db.add(rec)
        built += 1
        if built % 100 == 0:
            db.commit()

    db.commit()
    return {"built": built, "message": "Feature lake ready"}

@app.get("/features/{draw_number}")
def get_features(draw_number: int):
    db = next(get_db())
    rec = db.query(FeatureRecord).filter_by(draw_number=draw_number).first()
    if not rec:
        raise HTTPException(404, "Features not found for this draw")
    return json.loads(rec.features_json)

@app.get("/features/export/csv")
def export_features_csv():
    """Export full feature lake as CSV text."""
    from engine.features import FEATURE_COLS
    import csv, io
    db = next(get_db())
    recs = db.query(FeatureRecord).order_by(FeatureRecord.draw_number).all()
    output = io.StringIO()
    cols = ["draw_number"] + FEATURE_COLS
    writer = csv.DictWriter(output, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for rec in recs:
        row = json.loads(rec.features_json)
        writer.writerow({k: row.get(k,"") for k in cols})
    return {"csv": output.getvalue(), "rows": len(recs)}


# ── Backtest ───────────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    model_type: str = "baseline"   # baseline | rf | xgb | monte_carlo
    pool_size: int = 6             # 6 | 7 | 8
    n_preds: int = 10
    n_candidates: int = 150000
    min_history: int = 100
    mc_simulations: int = 5000

backtest_jobs = {}  # job_id -> status/result

@app.post("/backtest/run")
def start_backtest(req: BacktestRequest, background_tasks: BackgroundTasks):
    job_id = f"bt_{req.model_type}_{req.pool_size}_{int(time.time())}"
    backtest_jobs[job_id] = {"status": "running", "progress": 0, "result": None}

    def run_job():
        db = next(get_db())
        draws_raw = db.query(DrawRecord).order_by(DrawRecord.draw_number).all()
        draws = [r.to_dict() for r in draws_raw]

        def on_progress(done, total, dist, five_count):
            backtest_jobs[job_id]["progress"] = round(done/total*100)
            backtest_jobs[job_id]["five_plus_so_far"] = five_count

        result = run_backtest(
            draws=draws,
            model_type=req.model_type,
            pool_size=req.pool_size,
            n_preds=req.n_preds,
            n_candidates=req.n_candidates,
            min_history=req.min_history,
            mc_simulations=req.mc_simulations,
            progress_callback=on_progress,
        )

        # Save to DB
        br = BacktestRun(
            model_type=req.model_type,
            pool_size=req.pool_size,
            run_date=datetime.utcnow().isoformat(),
            total_tested=result["total_tested"],
            avg_match=result["avg_match"],
            rand_avg=result["rand_avg"],
            lift_pct=result["lift_pct"],
            three_plus_rate=result["three_plus_rate"],
            four_plus_rate=result["four_plus_rate"],
            five_plus_count=result["five_plus_count"],
            five_plus_rate=result["five_plus_rate"],
            result_json=json.dumps(result),
        )
        db.add(br)
        db.commit()
        backtest_jobs[job_id]["status"] = "done"
        backtest_jobs[job_id]["result"] = result
        backtest_jobs[job_id]["db_id"] = br.id

    background_tasks.add_task(run_job)
    return {"job_id": job_id, "status": "started"}

@app.get("/backtest/status/{job_id}")
def backtest_status(job_id: str):
    job = backtest_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job

@app.get("/backtest/results")
def list_backtest_results(limit: int = 20):
    db = next(get_db())
    rows = db.query(BacktestRun).order_by(BacktestRun.id.desc()).limit(limit).all()
    return [r.to_summary() for r in rows]

@app.get("/backtest/results/{result_id}")
def get_backtest_result(result_id: int):
    db = next(get_db())
    row = db.query(BacktestRun).filter_by(id=result_id).first()
    if not row:
        raise HTTPException(404, "Result not found")
    return json.loads(row.result_json)


# ── Prediction ─────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    pool_size: int = 6          # 6, 7, 8
    n_combos: int = 10
    model_type: str = "baseline"
    n_candidates: int = 200000
    use_monte_carlo: bool = False
    mc_simulations: int = 10000

@app.post("/predict")
def predict(req: PredictRequest):
    db = next(get_db())
    draws_raw = db.query(DrawRecord).order_by(DrawRecord.draw_number).all()
    if len(draws_raw) < 50:
        raise HTTPException(400, "Need at least 50 draws to predict")
    draws = [r.to_dict() for r in draws_raw]

    # Generate candidates
    candidates = generate_candidates(
        history=draws,
        pool_size=req.pool_size,
        n_candidates=req.n_candidates,
    )

    top = candidates[:req.n_combos * 5]

    if req.use_monte_carlo and top:
        mc = MonteCarloCandidateScorer(n_simulations=req.mc_simulations)
        mc_results = mc.score_candidates(
            [tuple(c) for c in top[:200]],
            draws[-100:]
        )
        combos_out = mc_results[:req.n_combos]
    else:
        combos_out = [{"combo": list(c), "score": None} for c in top[:req.n_combos]]

    # Save prediction
    pred = PredictionRecord(
        generated_date=datetime.utcnow().isoformat(),
        pool_size=req.pool_size,
        model_type=req.model_type,
        combos_json=json.dumps(combos_out),
        draw_count=len(draws),
    )
    db.add(pred)
    db.commit()

    return {
        "draw_count_used": len(draws),
        "pool_size": req.pool_size,
        "model_type": req.model_type,
        "predictions": combos_out,
        "prediction_id": pred.id,
    }

@app.get("/predict/history")
def prediction_history(limit: int = 20):
    db = next(get_db())
    rows = db.query(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(limit).all()
    return [r.to_summary() for r in rows]


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()
    print("StatLot 649 Engine started ✅")
