# Lead Persona Match + Deep Dive

This project runs a simple, reliable pipeline:

1) Select a contacts CSV (or set LEADS_CSV to skip the dialog)
2) Auto-load persona JSONs from the project root
3) Compute lightweight top-3 persona matches per candidate (token-free)
4) Optional deep-dive analysis via a single CrewAI task (LLM)

Outputs are written to `src/lead_score_flow/`:
- persona_top3.csv (lightweight matches)
- persona_deepdive_long.csv (all assessments)
- persona_deepdive_best.csv (one recommended persona per candidate)

The code is intentionally linear, with robust logging and provider safeguards for OpenRouter/OpenAI/Perplexity/Azure.

## Setup

Requirements: Python 3.11+ recommended; virtualenv suggested.

1) Install dependencies
```bash
pip install -U pip
pip install -r requirements.txt  # or: pip install crewai python-dotenv pydantic
```

2) Configure environment
Copy `.env.example` to `.env` and fill in your key(s). For OpenRouter:
```ini
OPENROUTER_API_KEY="your-openrouter-key"
LLM_MODEL="openrouter/openai/gpt-4o-mini"  # or omit to let fallback choose
USE_LLM_DEEPDIVE=1
```

Notes
- If `LLM_MODEL` is omitted, the app will choose a sensible default for your provider and try a few fallback models.
- You can also use `OPENAI_API_KEY`, `PERPLEXITY_API_KEY`, or `AZURE_OPENAI_API_KEY` instead of OpenRouter.
- `.env` is ignored by git. Do not commit secrets.

## Run

From the repo root:
```bash
python -u src/lead_score_flow/main.py
```
You’ll be prompted to select your contacts CSV unless you set `LEADS_CSV`.

## Troubleshooting

- If a model isn’t available on your account/plan, the app logs a warning and automatically tries fallback models.
- Deep-dive input snapshots are written under `src/lead_score_flow/_debug_payloads/` (ignored by git) to aid debugging.
- Set `LITELLM_DEBUG=1` for more provider diagnostics. Set `API_SANITY_CHECK=1` to print quick config tips at startup.

## What’s under the hood

- `src/lead_score_flow/main.py`: Orchestrates IO, matching, and deep dive. No event-driven flow or decorators.
- `src/lead_score_flow/crews/persona_pipeline_crew/`: Minimal CrewAI agent+task for the deep dive, with model normalization and fallbacks.
- `src/lead_score_flow/lead_types.py`: Pydantic models for inputs/outputs.

## Safety

`.gitignore` excludes `.env`, virtualenv, generated CSVs, and debug payloads. Review `.gitignore` before pushing.


