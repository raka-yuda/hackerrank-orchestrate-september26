# Final Full-Dataset Model Usage Report

Run type: hybrid OpenAI evidence extraction plus deterministic financial planning

Requests processed: 250

Output artifact: `output.csv`

Model provider and name: OpenAI / `gpt-4.1-mini-2025-04-14`

| Metric | Total | Per request |
|---|---:|---:|
| Model calls | 12 | 0.0480 |
| Input tokens | 26891 | 107.56 |
| Output tokens | 14295 | 57.18 |
| Thinking tokens (included in output) | 0 | 0.00 |
| Total tokens | 41186 | 164.74 |
| Generated-evidence cache hits | 0 | 0.0000 |
| Local OCR calls | 0 | 0.0000 |
| Model-to-local validation fallbacks | 0 | 0.0000 |
| Estimated model cost | USD 0.033628 | USD 0.00013451 |

This run used one model, so the table is both the per-model breakdown and the
overall total. Cost calculation: (26891 × 0.40 +
14295 × 1.60) / 1,000,000 = USD 0.0336284; divided
by 250 requests = USD 0.00013451 per request.

Token counts come from the selected provider's API usage metadata for calls made
during this run. The estimate uses USD 0.40 per million input tokens and
USD 1.60 per million output tokens, treating all input as uncached.
`OPENAI_INPUT_USD_PER_MILLION` and `OPENAI_OUTPUT_USD_PER_MILLION`
can override those rates. Actual billing may vary because of cached-input
discounts, credits, promotions, or negotiated pricing. The API key is read only
from the environment and is never written to this report or the evidence cache.
