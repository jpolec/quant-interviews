from __future__ import annotations

from math import exp

COEFFICIENTS = (0.35, -0.20, 0.10, 0.05)
INTERCEPT = -0.15
FEATURES = (1.2, -0.4, 0.8, 0.1)
MODEL_VERSION = "ch08-demo-v3"
FEATURE_SCHEMA_HASH = (
    "sha256:8afd31a62f52490379822fd497e6e00fec579eb9d5dab008d284827fa0616c6c"
)

score = INTERCEPT + sum(w * x for w, x in zip(COEFFICIENTS, FEATURES, strict=True))
probability = 1.0 / (1.0 + exp(-score))
print(
    f"{MODEL_VERSION} {FEATURE_SCHEMA_HASH} "
    f"probability={probability:.17g}"
)
