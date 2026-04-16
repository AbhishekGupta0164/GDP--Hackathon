from fastapi import APIRouter, UploadFile, Form, HTTPException
import pandas as pd
import numpy as np
import io, uuid

from core.fairness_metrics import compute_full_report

router = APIRouter()


def generate_predictions(df, target_col):
    y_true = df[target_col].values.astype(int)

    # simple simulation (same idea as your main.py)
    y_prob = np.random.uniform(0.3, 0.8, len(df))
    y_pred = (y_prob > 0.5).astype(int)

    return y_true, y_pred, y_prob


@router.post("/audit")
async def audit(
    file: UploadFile,
    domain: str = Form(...),
    sensitive_cols: str = Form(...),
    target_col: str = Form(...)
):
    try:
        df = pd.read_csv(file.file)

        sensitive_cols = sensitive_cols.split(",")

        if target_col not in df.columns:
            target_col = df.columns[-1]

        # 🔥 REQUIRED STEP
        y_true, y_pred, y_prob = generate_predictions(df, target_col)

        sensitive = df[sensitive_cols[0]].values

        # ✅ Correct function
        report = compute_full_report(y_true, y_pred, y_prob, sensitive)

        return {
            "run_id": str(uuid.uuid4())[:8],
            "domain": domain,
            "metrics": report.dict() if hasattr(report, "dict") else report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))