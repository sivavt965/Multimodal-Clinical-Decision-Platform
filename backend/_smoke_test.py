"""Quick smoke test for all backend endpoints."""
import urllib.request
import json

BASE = "http://127.0.0.1:8000"

def get(path):
    r = urllib.request.urlopen(f"{BASE}{path}")
    return json.loads(r.read())

# 1. Health
health = get("/api/health")
print(f"[1] /api/health => status={health['status']}, cases_loaded={health['cases_loaded']}")

# 2. GET /api/cases
cases = get("/api/cases")
print(f"\n[2] /api/cases => {len(cases)} summaries:")
for c in cases:
    print(f"    {c['case_id'][:12]}... | {c['patient_name']:15s} | Risk: {str(c['phase_a_risk_level']):8s} | Top: {c['top_finding_label']}")

# 3. GET /api/cases/{case_id} — pick the first case
case_id = cases[0]["case_id"]
detail = get(f"/api/cases/{case_id}")
print(f"\n[3] /api/cases/{case_id[:12]}... =>")
print(f"    Patient: {detail['patient']['first_name']} {detail['patient']['last_name']}")
print(f"    ECG HR:  {detail['case']['ecg_data']['heart_rate']} bpm")
print(f"    Labs Troponin: {detail['case']['lab_data']['troponin_ng_ml']} ng/mL")
print(f"    Predictions: {len(detail['predictions'])}")
for p in detail["predictions"]:
    print(f"      - {p['label']:20s}  prob={p['probability']:.2f}  badge={p['risk_badge']}")
print(f"    Consultation open: {detail['consultation']['is_open'] if detail.get('consultation') else 'N/A'}")

# 4. POST /api/consultation/{case_id}
msg = json.dumps({
    "id": "smoke-test-msg",
    "role": "ward_doctor",
    "type": "text",
    "content": "Smoke test message from _smoke_test.py",
    "sent_at": "2026-04-22T12:00:00Z",
    "read": False
}).encode()
req = urllib.request.Request(
    f"{BASE}/api/consultation/{case_id}",
    data=msg,
    headers={"Content-Type": "application/json"},
    method="POST",
)
resp = json.loads(urllib.request.urlopen(req).read())
print(f"\n[4] POST /api/consultation/{case_id[:12]}... => {resp}")

print("\n=== All endpoints OK ===")
