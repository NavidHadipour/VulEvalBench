# Setting up VulEvalBench on another machine

Written to be followed start to finish on a machine that has none of this
installed. Roughly 15 minutes, most of it waiting for a model download.

Tested on Windows 11 with Python 3.12. The paths below use Windows separators;
on Linux and macOS replace `.venv\Scripts\` with `.venv/bin/`.

---

## 1. What you need before starting

| Requirement | Why | Check |
|---|---|---|
| Python 3.10+ | Runs everything | `python --version` |
| ~6 GB free disk | Model weights (4.7 GB) plus dependencies | see step 3 |
| Git *(optional)* | Only for ingesting GitHub repositories | `git --version` |
| Access to an OpenAI-compatible LLM API | Judging, LLM detection, embeddings | see step 4 |

The judging API is the one external service this project cannot run without.
Everything else — generation, static analysis, ML — runs locally.

---

## 2. Install the Python dependencies

```bash
cd <project folder>
python -m venv .venv
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
```

Verify:

```bash
.venv\Scripts\bandit --version
.venv\Scripts\ruff --version
.venv\Scripts\semgrep --version
.venv\Scripts\python -c "import sklearn, flask, requests; print('ok')"
```

> **Semgrep version matters.** Releases before ~1.173 shipped no Windows
> executable — `pip install semgrep` reported success and installed nothing
> runnable. `requirements.txt` pins `semgrep>=1.173` for that reason. Semgrep
> also fetches its `p/python` ruleset from semgrep.dev on first use, so the
> first run needs network access.

---

## 3. Install Ollama and pull the generator model

Ollama runs the sample generator locally, so generation costs nothing per call.

**Windows**

```powershell
winget install Ollama.Ollama
```

**macOS / Linux** — download from <https://ollama.com/download>, or:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then pull a code model. **Check your free disk space first:**

```bash
ollama pull qwen2.5-coder:3b     # 1.9 GB -- the default; right for CPU-only
ollama pull qwen2.5-coder:7b     # 4.7 GB -- better, but needs a GPU to be practical
```

> **Pick by your hardware, not by model size.** Measured on a CPU-only machine
> (16 GB RAM, integrated graphics, no CUDA device):
>
> | model | per pair | memory | stability |
> |---|---|---|---|
> | `qwen2.5-coder:3b` | ~45 s | 1.9 GB | stable |
> | `qwen2.5-coder:7b` | ~457 s | 5.5 GB | intermittent OOM |
>
> The 7B model is ten times slower without GPU offload, and with only ~3 GB of
> RAM free it failed to allocate its KV cache repeatedly. A ten-pair run would
> take over an hour and might die partway.
>
> Check whether Ollama can use a GPU with `ollama ps` while a model is loaded:
> a `PROCESSOR` column reading `100% CPU` means no offload, and 7B will be
> painful. With a CUDA GPU, prefer 7B.

Whichever you pull, set it in `src/config.py` or via the environment:

```bash
set GENERATOR_MODEL=qwen2.5-coder:7b     # Windows
export GENERATOR_MODEL=qwen2.5-coder:7b  # Linux/macOS
```

Confirm the server is up and the model is present:

```bash
ollama list
curl http://localhost:11434/api/tags
```

If nothing answers on port 11434, start it with `ollama serve`. On Windows the
installer normally registers a background service that does this for you.

---

## 4. Supply the credentials

Two files in the project root. Both are listed in `.gitignore` and are never
copied into a Docker image. **Templates are provided** — copy and fill them in
rather than writing the files from scratch:

```bash
copy api-key-judgment.txt.example api-key-judgment.txt     # Windows
cp   api-key-judgment.txt.example api-key-judgment.txt     # Linux/macOS
```

### `api-key-judgment.txt` — required

Holds the base URL **and** key for the OpenAI-compatible API that serves the
judge models, the LLM detectors and the embedding model.

```
URL:
"https://llms.example.edu"

api key:
"sk-..."
```

The parser identifies values by shape, not by the label above them: anything
starting with `http` is the base URL, `sk-ant-` is an Anthropic key, any other
`sk-` is the API key. So the labels can be reworded freely and the two entries
can appear in either order.

> Both spellings — `api-key-judgment.txt` and `api-key-judgement.txt` — are
> accepted, because the project brief and the original file on disk disagreed.

### `api-key-llm.txt` — optional

Only needed to use Anthropic models as an extra detector arm.

```
anthropic api key:
"sk-ant-api03-..."
```

### Pointing at a different API

Any OpenAI-compatible endpoint works — vLLM, LiteLLM, Ollama's own compatible
API, or a commercial provider. Put its base URL in the file, then update the
model ids in `src/config.py`:

```python
JUDGMENT_MODELS = {
    "<model-id-from-/v1/models>": "Display name",
}
DEFAULT_JUDGE_A     = "<model-id>"
DEFAULT_JUDGE_B     = "<another-model-id>"
DEFAULT_LLM_DETECTOR = "<a-third-model-id>"
EMBEDDING_MODEL     = "<an-embedding-model-id>"
```

List what an endpoint offers with:

```bash
curl -H "Authorization: Bearer sk-..." https://your-endpoint/v1/models
```

The launch page shows a green dot for each credential it found, so a
misconfigured file is visible before you start a run rather than after.

---

## 5. Check the whole environment

```bash
.venv\Scripts\python src\check_setup.py
```

This checks every prerequisite — Python version, packages, the three static
analysers, both credential files, the Ollama server and model, the CWE
hierarchy cache, and that `runs/` and `logs/` are writable — and prints a
checklist. Every `[FAIL]` line says exactly what to run to fix it:

```
  [ ok ]  Python                      3.12.3
  [ ok ]  tool: semgrep               ...\.venv\Scripts\semgrep.exe
  [FAIL]  Ollama model                'qwen2.5-coder:3b' not pulled -- run: ollama pull qwen2.5-coder:3b
  ...
  1 required check(s) failed -- fix the [FAIL] lines above.
```

Exit code is 0 only when everything required passes, so it also works as a
gate in a script or CI job. Re-run it after each fix until it reports
`All required checks passed`.

---

## 6. Run it

**Web interface** — the normal way:

```bash
.venv\Scripts\python web\app.py
```

Open <http://localhost:8000>. Configuration is in the left sidebar; press
**Run pipeline**.

**Command line** — the same code, stage by stage:

```bash
.venv\Scripts\python src\generate.py            # prints a run id
.venv\Scripts\python src\judge.py    <run_id>
.venv\Scripts\python src\detect.py   <run_id>
.venv\Scripts\python src\evaluate.py <run_id>
```

Each script falls back to the most recent run when given no id, so for a
straight-through run you can omit the argument every time.

---

## 7. Docker

```bash
docker compose up --build
```

Then open <http://localhost:8000>.

Two things to know:

- **Ollama is not in the container.** It runs on the host, and the container
  reaches it through `host.docker.internal`. Set `OLLAMA_BASE_URL` accordingly
  in `docker-compose.yml`, and make sure Ollama is listening on all interfaces
  (`OLLAMA_HOST=0.0.0.0`) rather than only on loopback.
- **`runs/` and `logs/` are bind-mounted**, so results survive
  `docker compose down`. Credential files are mounted read-only at run time and
  never baked into an image layer — a secret in a layer persists in the image
  history even if a later layer deletes it.

---

## Troubleshooting

**"Could not reach Ollama"** — the server is not running. `ollama serve`, then
check `curl http://localhost:11434/api/tags`.

**"No judgment key file found"** — `api-key-judgment.txt` is missing from the
project root, or its values are not in double quotes.

**Generation produces no usable pairs** — almost always the model tag in
`config.GENERATOR_MODEL` is not one that `ollama list` shows. Check
`logs/<run_id>.log` for the per-pair rejection reasons.

**Every sample is dropped in judging** — expected with a small generator model.
Look at `runs/<run_id>/02_judgement/judgement_dropped.json`, which records the
failed rule for each sample. Either move to a larger generator, or relax
`JUDGE_CONSENSUS` from `"all"` to `"any"` in `config.py`.

**The ML arm is skipped with "only one class is present"** — judging admitted
samples of only one label, so there is nothing to classify. Generate more
pairs, or use a stronger generator.

**"not enough space on the disk" during `ollama pull`** — a failed pull leaves
a `-partial` blob behind that still occupies the space. Delete
`~/.ollama/models/blobs/*-partial*` and pull a smaller tag.

**Windows console `UnicodeEncodeError`** — the reports are deliberately plain
ASCII to avoid this. If you add output, keep it ASCII or set
`PYTHONIOENCODING=utf-8`.
