# Buy or Wait? — Hybrid Financial Decision Agent

This submission uses OpenAI or Gemini for schema-constrained multimodal evidence
extraction when a matching key is available, with a deterministic local fallback.
Financial-state reconstruction, cash-flow simulation, plan selection, and final
validation always remain deterministic. No third-party Python package is required.

## Run

From the repository root:

```bash
python3 code/main.py
```

The command reads the participant-facing files in `dataset/`, validates every
decision, writes `output.csv`, and refreshes `evaluation/usage_report.md`. With no
key, `auto` uses local Tesseract OCR and multilingual rules.

To test OpenAI, set the key in your shell—never in source code or chat—and score
the public examples first:

```bash
export OPENAI_API_KEY="your-key-from-the-OpenAI-platform"
python3 code/evaluation/main.py --evidence-provider openai
```

Then run the full dataset only after reviewing the public-example result:

```bash
python3 code/main.py --evidence-provider openai
```

If `code.zip` was extracted directly beside `dataset/`, run `python3 main.py`
instead. Both layouts are detected automatically.

Gemini remains available as an alternative:

```bash
export GEMINI_API_KEY="your-key-from-Google-AI-Studio"
python3 code/evaluation/main.py --evidence-provider gemini
```

If those results are acceptable, run the full dataset with Gemini:

```bash
python3 code/main.py --evidence-provider gemini
```

Responses are cached by input content under `.cache/buy_or_wait/`, so a retry does
not repeatedly spend quota. Delete that generated cache only when intentionally
performing a fresh final usage-measurement run.

Hosted-provider mode sends relevant message text and image bytes to the selected
provider's API. Use `--evidence-provider local` when evidence must remain entirely
on the machine. `auto` prefers OpenAI when `OPENAI_API_KEY` is present, then
Gemini, then the local fallback.

Optional arguments:

```bash
python3 code/main.py --dataset dataset --requests requests.csv --output output.csv \
  --evidence-provider openai --openai-model gpt-4.1-mini-2025-04-14
```

Run the public-example evaluator and offline tests with:

```bash
python3 code/evaluation/main.py
python3 -m unittest discover -s code/tests -v
```

Python 3.10 or newer is recommended. Tesseract is required only for the no-key
fallback or when a Gemini response fails validation. `GEMINI_MIN_INTERVAL_SECONDS`
defaults to `4.1` to be gentle with free-tier rate limits. Optional
`GEMINI_INPUT_USD_PER_MILLION` and `GEMINI_OUTPUT_USD_PER_MILLION` values let the
usage report calculate paid-tier cost; both default to zero for free-tier runs.
The extraction calls use `GEMINI_THINKING_LEVEL=low` so structured output has
enough room and does not spend unnecessary reasoning tokens. OpenAI mode uses the
Responses API with `store=false`, strict JSON Schema output, and
`OPENAI_MIN_INTERVAL_SECONDS=1.0` by default. The default model is pinned to
`gpt-4.1-mini-2025-04-14` so reruns do not silently change behavior. Optional
`OPENAI_INPUT_USD_PER_MILLION` and `OPENAI_OUTPUT_USD_PER_MILLION` values provide
the corresponding cost estimate. They default to the pinned model's standard
USD 0.40 input and USD 1.60 output rates per million tokens; override them only
when a different pricing contract applies.

## Approach

1. Load and validate profiles, requests, events, fixed exchange rates, payment
   options, messages, and image links with `Decimal` monetary arithmetic.
2. Resolve only relevant blank event amounts and message amendments through one
   evidence interface. OpenAI or Gemini handles images and batched multilingual
   messages; local OCR/rules are the fallback. All extracted facts are schema,
   currency, confidence, date, and source-grounding checked.
3. Infer recurring streams only from stable dated history, reserve pending and
   scheduled debits, count settled or confirmed income on its cash date, and
   simulate every balance change over 90 days.
4. Evaluate full payment, waiting, exactly-two-part partial payment, supplied
   installment schedules, and up to three eligible spending changes.
5. Rank safe plans using the challenge rules and run deterministic validation
   before emitting the exact required CSV schema.

No organizer-only file or evaluation label is read by `code/main.py`. Source code
contains no image hash-to-answer table or request-specific prediction labels;
SHA-256 is used only to invalidate generated cache entries when evidence changes.
