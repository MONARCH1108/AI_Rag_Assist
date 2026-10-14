import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

client = Groq(api_key=api_key, timeout=30.0, max_retries=0)

print("\nFetching available Groq models...")

models = client.models.list()
results = []

print(f"Models returned: {len(models.data)}\n")

for model in models.data:
    model_id = model.id
    print(f"Testing: {model_id}")

    result = {
        "model_id": model_id,
        "active": getattr(model, "active", None),
        "context_window": getattr(model, "context_window", None),
        "status": None,
        "error": None,
        "latency_seconds": None,
    }

    start = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: OK",
                }
            ],
            max_completion_tokens=10,
        )

        result["status"] = "working"
        result["response"] = response.choices[0].message.content
        print("  -> Chat completion successful")

    except Exception as exc:
        status_code = getattr(exc, "status_code", None)

        result["error"] = str(exc)
        result["http_status"] = status_code

        if status_code == 429:
            result["status"] = "rate_limited"
        elif status_code in (401, 403):
            result["status"] = "access_denied"
        elif status_code == 404:
            result["status"] = "unavailable_or_unsupported"
        elif status_code in (400, 422):
            result["status"] = "unsupported_or_invalid_request"
        else:
            result["status"] = "error"

        print(f"  -> {result['status']}: {exc}")

    result["latency_seconds"] = round(
        time.perf_counter() - start, 3
    )
    results.append(result)

    # Avoid rapidly sending requests to the API.
    time.sleep(1)

report = {
    "tested_at": datetime.now(timezone.utc).isoformat(),
    "models_discovered": len(models.data),
    "models_working": sum(
        r["status"] == "working" for r in results
    ),
    "results": results,
}

output = Path("reports/groq_model_results.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n========== SUMMARY ==========")
print(f"Discovered: {report['models_discovered']}")
print(f"Working:    {report['models_working']}")
print(f"Report:     {output}")
