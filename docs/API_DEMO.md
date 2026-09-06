# AlphaLDM Discovery Console

FastAPI serves the API and its self-contained dashboard. Choose **Mock** for
measured historical replay or **Online** for a real DeepSeek V4 Flash run.
The frontend needs no Node build, CDN, or internet connection for historical playback.

## Start on this Mac

```bash
cd "/Users/senorita/Desktop/Alpha Mining Report/AlphaResearch"
conda activate AlphaResearch
pip install -e '.[dev,api]'
bash scripts/start_api.sh
```

Open <http://127.0.0.1:8765>. Interactive API documentation is at
<http://127.0.0.1:8765/docs>. The optional first launcher argument changes the port.
`alphaldm-api --port 8765` is the equivalent Python entry point.

The launcher keeps macOS awake while the server runs. Each live worker also
holds its own `caffeinate` assertion until it finishes or pauses. Closing the
browser or restarting the API does not stop an active research worker.

## Mock: measured historical replay

Select seed **42**, **123**, or **456**, choose a playback interval, and click
**开始回放**. The dashboard reveals R0, R5, …, R100, with an adjustable delay
between frames. Each seed has 42 initial factors plus 300 newly evaluated
factors. The curve, leading factor, selected Top-5, and factor counts are
limited to the currently revealed round.

This mode reads `demo/replays/seed<seed>.json`. It never invokes a proposer,
profiler, market evaluator, or paid LLM. Pause/resume, refresh/reconnect, recent
sessions, formula copying, and JSON export are supported.

The original snapshots had ten-round checkpoints plus R38/R75. The missing
five-round checkpoints were prepared separately with the existing Validation
selector and real evaluator. Identical pools reuse their previously measured
results for the same period; different pools receive a real evaluation. No
point is interpolated. The preparation process records source-file SHA-256
hashes, keeps full report/provenance files under `runtime/api/replay_build/`,
and leaves the original `runs/ldm_continuous_discovery/` files unchanged.

To rebuild from the preserved results, with the real evaluator available:

```bash
python scripts/prepare_api_replays.py --seeds 42 123 456
```

Full supplementary five-round checkpoint evidence is preserved as compressed
JSON under `demo/replay_audits/` when the demo is packaged. Replay JSON also
contains the ordered factor inventory and all 21 display frames.

## Online: real discovery

Online uses the existing DeepSeek V4 Flash configuration. It retains the
12D representation, scalar GP/UCB, 8 proposals per batch, 3 verified Train
evaluations per round, split dates, and 0.8 Validation correlation filter.
Only the checkpoint schedule is changed to **every five rounds**.

The live search needs the local Qlib data, the AlphaBench FFO service, and the
LLM credential. The dashboard checks these prerequisites before launch and
reports what is unavailable. The Mac credential is read from the existing
`AlphaResearch-LiteLLM` Keychain entry, or from `ALPHARESEARCH_LLM_API_KEY`.
No credential is accepted in an API request or returned in a response.

If the evaluator is stopped:

```bash
export QLIB_DATA_PATH="$PWD/../quantaalpha_qlib_csi300/cn_data"
export QLIB_PROVIDER_URI="$QLIB_DATA_PATH"
export FFO_QLIB_DATA_PATH="$QLIB_DATA_PATH"
ppo start backend
```

Optional server-side overrides:

- `ALPHARESEARCH_QLIB_PROVIDER_URI`: local provider directory;
- `ALPHARESEARCH_FFO_URL`: FFO service URL.

Select **Online**, a seed, and a target (a multiple of 5), then click **启动真实搜索**.
This starts real API usage. The job first evaluates the 42 initial factors and
reports R0. After each committed round the current best factor and saved-factor
count can update; curves gain another point at each five-round checkpoint.
The pulse panel identifies proposal, profiling, GP/UCB, Train, Validation,
Test, and checkpoint stages. The duration is computational time, not the
artificial replay interval.

**暂停** requests a safe pause after the current complete round and its scheduled
report. It can take time while an evaluation or LLM request finishes. **继续**
restores the same run. After completion, increase the target and click **续跑**
to extend it, e.g. R100 → R200. Only one online worker runs at a time to avoid
CPU contention and duplicate paid jobs; mock playback can run alongside it.

## API examples

Create a replay; replace `mock` with `online` for a real run:

```bash
curl -X POST http://127.0.0.1:8765/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"mode":"mock","seed":42,"target_rounds":100,"interval_seconds":2.5}'
```

The response is `202 Accepted` with an `id`. The dashboard polls the following
snapshot endpoint once per second; other clients may use the same interface.

```bash
curl http://127.0.0.1:8765/api/runs/RUN_ID
curl http://127.0.0.1:8765/api/runs/RUN_ID/factors
curl http://127.0.0.1:8765/api/runs/RUN_ID/export
curl -X POST http://127.0.0.1:8765/api/runs/RUN_ID/pause
curl -X POST http://127.0.0.1:8765/api/runs/RUN_ID/resume \
  -H 'Content-Type: application/json' -d '{"target_rounds":200}'
```

Other endpoints: `GET /api/health`, `/api/readiness`, `/api/archives`, `/api/runs`.
The factor endpoint accepts `offset` and `limit`, preserves evaluation order,
and returns only factors available at the current committed/revealed round.
Export includes the current curves, selected Top-5, ordered factors, and state.

## Result interpretation

- **Train**: best single-factor Train RankIC among committed factors.
- **Test (solid)**: the actual Test RankIC of each Validation-selected Top-5.
- **Test best-so-far (dashed, optional)**: a retrospective envelope of those
  Test measurements. It is never used to select a factor, a checkpoint, or a
  future proposal.
- **0.035**: the existing diagnostic recording threshold.

Real Test values may decline. Failed measurements remain unavailable; they are
never replaced by zero. A run is marked completed only after the target round
and all required measured reports are present. A report failure preserves the
search checkpoint and is retried during final reporting or explicit resume.

“Online” means a live computation using the existing fixed Train/Validation/Test
protocol. It does not imply a live market feed or a rolling train/test split.

## Storage and deployment

```
demo/replays/                         # Small, committed historical replay data
demo/replay_audits/                   # Compressed measurement evidence
src/alpha_research/api/static/        # Bundled dashboard (HTML/CSS/JavaScript)
runtime/api/jobs/<id>/
  job.json                           # Safe launch parameters, no credentials
  state.json, snapshot.json           # Atomic progress and dashboard state
  control.json, process.json          # Safe pause control and worker identity
  timeline.json, worker.log
  run/<canonical-run-name>/          # Online ledger, sequence, reports, RNG state
runtime/api/replay_build/             # Full reporting-only backfill workspaces
```

API jobs are isolated from the existing experiment folders. Runtime files are
gitignored; back them up separately after a live demonstration. Source/results
on `main` are unaffected by work on `AlphaLDM_API`.

Run a **single** Uvicorn worker. The service and online worker use separate
process locks, and stored PID creation times prevent mistaking a reused PID
for the original worker. If a worker dies, the API shows `interrupted` and
offers explicit resume; it does not silently start another paid run.

The launcher binds to localhost. For remote deployment, use the same repository
and data on a host with Python 3.11, place authenticated HTTPS access in front
of the service, and configure `ALPHALDM_API_ALLOWED_HOSTS` for the chosen host.
The standalone dashboard has no account system. Keep the service private until
that access layer is configured. The original AlphaBench submodule and its FFO
overlays are required for real evaluation; follow the repository installation
instructions when cloning on another machine.

## Verification

```bash
pytest tests/test_api.py tests/test_ldm_continuous_discovery.py tests/test_ldm_financial_plot.py -q
```

Tests cover measured five-round replay provenance/visibility, negative and
missing Test values, job pause/resume across service lifetimes, request bounds,
cross-origin controls, credential exclusion, and exact round continuation.
The adapter test uses explicitly synthetic dependencies and is not scientific
evidence. Actual online smoke results live in their own API job directory.

FastAPI lifecycle and static-file integration follow its
[lifespan](https://fastapi.tiangolo.com/advanced/events/) and
[static files](https://fastapi.tiangolo.com/tutorial/static-files/) documentation.
