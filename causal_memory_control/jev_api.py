"""TypeSafe Jev HTTP adapter, following the protocol supplied by the user."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class JevAPIError(RuntimeError):
    pass


def judge(request):
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise JevAPIError("TYPESAFE_API_KEY is not set")
    provider = request.get("provider", {})
    model = provider.get("model_id", "jev-1.13.0")
    endpoint = provider.get("endpoint", "https://api.typesafe.ai/v1/systemone")
    if not endpoint.startswith("https://"):
        raise JevAPIError("Jev API endpoint must use HTTPS")
    payload = {"model": model, "state": json.loads(request["state"]),
               "questions": {name:{"type":"noul", "instructions":instructions}
                             for name,instructions in request["questions"].items()}}
    http_request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), method="POST",
                           headers={"Authorization":f"Bearer {key}", "Content-Type":"application/json"})
    start = time.monotonic()
    response = None
    for attempt in range(3):
        try:
            with urlopen(http_request, timeout=60) as handle:
                response = json.loads(handle.read())
            break
        except HTTPError as exc:
            # Do not log server bodies or request headers, which could echo credentials.
            status = exc.code
            exc.close()
            if attempt<2 and (status==429 or status>=500):
                time.sleep(2**attempt)
                continue
            raise JevAPIError(f"Jev HTTP {status} from configured endpoint") from None
        except (URLError, TimeoutError, OSError) as exc:
            if attempt<2:
                time.sleep(2**attempt)
                continue
            raise JevAPIError(f"Jev network request failed ({type(exc).__name__})") from None
        except (ValueError, UnicodeDecodeError):
            raise JevAPIError("Jev returned invalid JSON") from None
    if not isinstance(response, Mapping) or not isinstance(response.get("answers"), Mapping):
        raise JevAPIError("Jev response has no answers object")
    if response.get("model") not in (None, model):
        raise JevAPIError("Jev returned a different model version")
    probabilities = {}
    for name in request["questions"]:
        answer = response["answers"].get(name)
        value = answer.get("noul") if isinstance(answer, Mapping) else None
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise JevAPIError(f"Jev answers.{name}.noul must be a probability in [0,1]")
        probabilities[name] = float(value)
    usage = response.get("usage", {})
    return {"probabilities": probabilities, "usage": usage if isinstance(usage,Mapping) else {},
            "model": model, "request_seconds": time.monotonic()-start}
