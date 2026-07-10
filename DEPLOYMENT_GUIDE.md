# Deployment Guide — Getting a Public Demo Link for the Paper

Goal: expose your running Hybrid GraphRAG system at a public URL (e.g.
`https://legal-rag.<something>.trycloudflare.com`) so the professor can open it,
then paste that URL into the abstract in place of
`https://REPLACE-WITH-YOUR-DEMO-URL`.

**Important reality check.** Your system runs a **Qwen3-32B-AWQ** model on
**vLLM**, a **Neo4j** graph (804k nodes / 1.97M edges), and a **Chroma** vector
store (68,835 chunks). I cannot host this from here — it needs your GPU and your
local databases. This guide is the fastest path to a link **from your own
machine**. The public URL will only be live while your machine and the tunnel
are running, which is normal and fine for a thesis defense demo.

---

## 0. Prerequisites (already on your machine)

| Component | Role | Port |
|---|---|---|
| Neo4j | Legal knowledge graph | 7687 |
| Chroma | Dense vector store (local files in `data/`) | — |
| vLLM (OpenAI-compatible) | Serves `Qwen3-32B-AWQ` for generation + Stage-2 | 8001 (example) |
| FastAPI (`api/main.py`) | Backend API | 8000 |
| Next.js (`frontend/`) | Chat UI | 3000 |

GPU note: `Qwen3-32B-AWQ` needs roughly **24 GB+ VRAM** (a single 3090/4090/A5000
works with `--gpu-memory-utilization` tuned; two GPUs are more comfortable).

---

## 1. Start the backend stack locally

Run each in its own terminal from the project root.

**a) Neo4j** — start your existing Neo4j instance (Desktop or service). Confirm
it listens on `neo4j://127.0.0.1:7687` and that credentials match `.env` /
`configs/config.yaml`.

**b) vLLM (LLM endpoint)** — start the OpenAI-compatible server, for example:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B-AWQ \
  --quantization awq \
  --port 8001 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192
```

Then point the app at it in `.env`:

```
OPENAI_API_KEY=not-required
OPENAI_BASE_URL=http://localhost:8001/v1
OPENAI_MODEL=Qwen/Qwen3-32B-AWQ
```

**c) FastAPI backend**:

```bash
.venv\Scripts\activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Sanity check: open `http://localhost:8000/` → should return
`{"message": "Vietnamese Legal QA API is running", ...}`. The interactive API
docs are at `http://localhost:8000/docs`.

---

## 2. Fix the one thing that blocks a shareable link

The frontend currently calls `http://localhost:8000` **hard-coded**
(`frontend/src/app/page.tsx`). That works on your machine but breaks for a
remote visitor, because their browser would try to reach *their own* localhost.

The clean fix is to serve the API and the UI from **one origin** using a Next.js
rewrite, so you only need to tunnel one port.

**a) Add a rewrite in `frontend/next.config.ts`:**

```ts
/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: "http://localhost:8000/:path*" },
    ];
  },
};
export default nextConfig;
```

**b) Point the frontend at the relative path.** In
`frontend/src/app/page.tsx`, replace every `http://localhost:8000` with `/api`
(e.g. `http://localhost:8000/query_stream` → `/api/query_stream`). A quick
find-and-replace on that one string is enough (7 occurrences).

**c) Build and start the frontend:**

```bash
cd frontend
npm install
npm run build
npm run start   # serves on http://localhost:3000
```

Now `http://localhost:3000` serves both the UI and (via `/api/...`) the backend
from a single port — the only port you need to expose.

> If you prefer not to edit the frontend, you can instead tunnel port **8000**
> and demo through the Swagger UI at `/docs`. It's less pretty but requires zero
> code changes and is a legitimate defense artifact.

---

## 3. Expose it publicly (pick one)

### Option A — Cloudflare Tunnel (recommended, free, no signup for quick tunnels)

```bash
# install once: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
cloudflared tunnel --url http://localhost:3000
```

It prints a URL like `https://legal-rag-xxxx.trycloudflare.com`. That is your
demo link.

### Option B — ngrok

```bash
ngrok http 3000
```

Copy the `https://<random>.ngrok-free.app` forwarding URL.

For a **stable** URL that survives restarts (nicer for a submitted paper), use a
Cloudflare **named tunnel** with your own domain, or an ngrok reserved domain
(paid). A quick tunnel URL changes every restart.

---

## 4. Put the link in the paper

Open `paper/manuscript_english.tex`, find the last line of the abstract:

```latex
\url{https://REPLACE-WITH-YOUR-DEMO-URL}.}
```

Replace `REPLACE-WITH-YOUR-DEMO-URL` with your actual URL, e.g.:

```latex
\url{https://legal-rag-xxxx.trycloudflare.com}.}
```

Recompile: `pdflatex → bibtex → pdflatex → pdflatex`.

---

## 5. Stability tips for defense day

- Keep the vLLM, FastAPI, and tunnel terminals open for the whole session.
- Warm up the pipeline once before the demo (the first query loads the model and
  the graph/vector clients, which is slow).
- If the professor only needs to *see it works* rather than keep a permanent
  link, a `trycloudflare.com` URL generated the morning of the defense is the
  least-effort option.
- For a permanently reachable demo, deploy the backend to a GPU cloud VM
  (RunPod, Lambda, Vast.ai) and the frontend to the same VM; then a named
  Cloudflare tunnel gives a fixed HTTPS URL you can cite in the final PDF.

---

## Quick checklist

- [ ] Neo4j up on 7687
- [ ] vLLM serving `Qwen3-32B-AWQ` on 8001, `.env` points to it
- [ ] FastAPI up on 8000 (`/` returns ok)
- [ ] `next.config.ts` rewrite added + `page.tsx` uses `/api`
- [ ] `npm run build && npm run start` → UI on 3000 works locally
- [ ] `cloudflared`/`ngrok` tunnel on 3000 → public URL opens from your phone
- [ ] URL pasted into the abstract, PDF recompiled
