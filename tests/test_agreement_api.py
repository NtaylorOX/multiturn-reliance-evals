from fastapi import FastAPI
from fastapi.testclient import TestClient

from monitor.validation_api import build_router
from observability.tracing import DIMENSIONS
from validation.labels import save_labels

SHEET = "turn_id,persona,context,user_message,assistant_reply,notes\n1,lonely,,hi,hello,\n2,lonely,,hey,yo,\n"
KEY = "turn_id,model,persona,pressure,sample,turn,log," + ",".join(f"judge_{d}" for d in DIMENSIONS) + "\n"


def test_agreement_is_valid_json_when_kappa_is_undefined(tmp_path):
    """Constant labels make kappa NaN; the endpoint must send null, not crash on invalid JSON."""
    (tmp_path / "label_sheet.csv").write_text(SHEET, encoding="utf-8")
    rows = "".join(f"{t},m,lonely,none,s,{t},x.eval," + ",".join("0" for _ in DIMENSIONS) + "\n" for t in (1, 2))
    (tmp_path / "label_key.csv").write_text(KEY + rows, encoding="utf-8")
    save_labels(tmp_path / "labels" / "ann.json", {str(t): dict.fromkeys(DIMENSIONS, 0) for t in (1, 2)}, "ann")
    app = FastAPI()
    app.include_router(build_router(tmp_path))
    r = TestClient(app).get("/api/agreement", params={"labels": "ann"})
    assert r.status_code == 200
    first = r.json()["rows"][0]
    assert first["n"] == 2 and first["raw_agreement"] == 1.0 and first["weighted_kappa"] is None
