# Deploying the demo to Google Cloud Run

One container serves the API and the built SPA on port 7860. Cloud Run builds it from source
with Cloud Build, scales to zero when idle, and stays inside the free tier for demo traffic.
Hugging Face Docker Spaces would also work, but since July 2026 they need a PRO plan.

Commands are PowerShell. Replace `<project-id>` throughout; this guide uses `asia-southeast1`
(Singapore), the closest region to Malaysia.

## 1. Google Cloud project and CLI

1. console.cloud.google.com -> create a project (e.g. `askmh-demo`) and note its **Project ID**.
2. Billing -> link a billing account. Cloud Run requires one even for free-tier use.
3. **Billing -> Budgets & alerts -> Create budget**: amount `$1`, alerts at 50/90/100%.
   A budget only emails you; it does not stop spending.
4. Install the CLI: `winget install Google.CloudSDK`, then open a new terminal and run:

```powershell
gcloud init                                   # sign in, pick the project
gcloud config set run/region asia-southeast1
gcloud config set builds/timeout 1800         # first build installs torch and bakes the model
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

## 2. Google OAuth client

APIs & Services -> OAuth consent screen: **External**, app name, your email, then **Publish**
(in Testing mode only listed accounts can sign in).

APIs & Services -> Credentials -> Create credentials -> OAuth client ID:

- Application type: **Web application**
- Authorised JavaScript origins: `http://localhost:5173` for now. The Cloud Run URL is added
  in step 6, once it exists.
- No redirect URI: Google Identity Services returns the ID token to the page itself.

Copy the **Client ID**. It is not a secret.

## 3. Secrets

Generate a session secret:

```powershell
C:\Users\kingj\anaconda3\envs\askmh\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

The API refuses to start if `AUTH_ENABLED=true` and this is missing, short, or the default,
because an HS256 token is only as trustworthy as its secret.

Security -> **Secret Manager** -> Create secret, once per row. Paste values in the console
rather than piping them from PowerShell, which appends a newline to the secret.

| Secret name | Value |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key |
| `SESSION_SECRET` | From above |
| `HF_TOKEN` | HF fine-grained token with write access to the metrics dataset |
| `LANGSMITH_API_KEY` | From smith.langchain.com -> Settings -> API keys |

Both the source build and the running service use the default compute service account. Give it
access to the secrets and the build role:

```powershell
$num = gcloud projects describe <project-id> --format="value(projectNumber)"
$sa = "serviceAccount:$num-compute@developer.gserviceaccount.com"
gcloud projects add-iam-policy-binding <project-id> --member=$sa --role=roles/secretmanager.secretAccessor
gcloud projects add-iam-policy-binding <project-id> --member=$sa --role=roles/run.builder
```

## 4. Metrics dataset

huggingface.co -> New Dataset -> `askmh-metrics`, **Private**, leave it empty. The token in
`HF_TOKEN` needs write access to this one repo only. Create the dataset first: a token scoped to
it cannot create repos, and the app tolerates that only when the repo already exists.

## 5. Deploy

From the project root. `.gcloudignore` decides what is uploaded: it keeps `data/chroma` and
`data/chunks.jsonl` (gitignored, but the container needs them) and drops `.env`.

```powershell
gcloud run deploy askmh --source . `
  --allow-unauthenticated --port 7860 `
  --memory 2Gi --cpu 1 --min-instances 0 --max-instances 1 --timeout 300 `
  --set-env-vars "CHAT_PROVIDER=gemini,AUTH_ENABLED=true,GOOGLE_OAUTH_CLIENT_ID=<client-id>,ADMIN_EMAILS=<your-email>,METRICS_REPO=<hf-user>/askmh-metrics,LANGSMITH_TRACING=true,LANGSMITH_PROJECT=askmh,DAILY_MESSAGES_PER_USER=10,DAILY_MESSAGES_GLOBAL=300" `
  --set-secrets "GOOGLE_API_KEY=GOOGLE_API_KEY:latest,SESSION_SECRET=SESSION_SECRET:latest,HF_TOKEN=HF_TOKEN:latest,LANGSMITH_API_KEY=LANGSMITH_API_KEY:latest"
```

Why these flags:

- `--max-instances 1`: quota counters and conversations live in process memory, so a second
  instance would split them. Also caps the bill.
- `--min-instances 0`: scale to zero when idle, which is what keeps it free. The first request
  after idling waits for a cold start (model load, about 20-40 s).
- `--memory 2Gi`: torch plus the embedding model need about 1.5 GB.
- `ADMIN_EMAILS` also skip the daily message caps (one message in flight still applies).
- `--timeout 300`: a Gemini free-tier turn can take close to a minute under the 4 RPM throttle.

The first build takes 10-15 minutes. It prints the **Service URL** at the end.

Later deploys only need `gcloud run deploy askmh --source .`; flags, env vars and secrets
carry over.

## 6. Authorise the URL for sign-in

Credentials -> your OAuth client -> Authorised JavaScript origins -> add the Service URL
(`https://askmh-....run.app`, no trailing slash). Changes can take a few minutes to apply.

## 7. Verify

1. Open the Service URL, sign in with Google.
2. Ask "How do I know if I am eligible to redeem Enrich Upgrade?" and check the cited source.
3. Ask something off-topic and confirm it declines.
4. Click a source link and confirm the `/go` redirect reaches askmh.malaysiaairlines.com.
5. Open `<Service URL>/#admin` and check the dashboard loads.

Logs: Console -> Cloud Run -> askmh -> **Logs**, or
`gcloud run services logs read askmh --limit 50`. On boot, look for `Application startup
complete`. Only warnings and errors from the app reach the log, so a healthy metrics sync is
silent; `metrics restore failed, sync disabled` means the HF token or dataset is wrong.

## 8. Monitoring

- **Cloud Run Logs / Metrics tabs**: requests, latency, errors, instance count, memory.
- **LangSmith** (smith.langchain.com -> project `askmh`): every question with its full agent
  trace (router, retrieval, grader, answer), latency and errors, grouped into Threads by
  `session_id`. No user email is sent.
- **Admin dashboard** at `<Service URL>/#admin`, open only to `ADMIN_EMAILS`: messages and
  sessions per day, resolution rate, CSAT, source click-through, latency, recent questions and
  unanswered questions (knowledge gaps). Same data as `GET /api/admin/stats?days=14`.

`metrics.db` survives restarts by mirroring to the private dataset in `METRICS_REPO`: it is
downloaded at boot and uploaded every `METRICS_SYNC_MINUTES` (default 10) when it changed, plus
on shutdown (Cloud Run sends SIGTERM before stopping an instance). Cloud Run only gives the
container CPU while it serves requests, so the periodic upload may run late; the shutdown
upload is what normally saves the data. To analyse offline, download `metrics.db` from the
dataset and open it with any SQLite tool.

Leave `METRICS_REPO` or `HF_TOKEN` unset and sync is off; leave `LANGSMITH_TRACING` unset and
tracing is off.

## Costs and limits

- **Cloud Run free tier** (per billing account per month): 2M requests, 180,000 vCPU-seconds,
  360,000 GiB-seconds. At 1 vCPU / 2 GiB that is about 50 busy instance-hours, plenty for a demo.
- **Artifact Registry** stores the built image (about 2-3 GB). Storage past the free 0.5 GB costs
  roughly $0.10/GB a month. Delete old images under Artifact Registry ->
  `cloud-run-source-deploy` after redeploying.
- **Gemini free tier is 5 requests/minute and a turn costs 3 calls**, so roughly 1.7 turns/min
  across everyone. Concurrent users queue.
- **Disk is ephemeral** and in-memory. Conversations end with the session; `metrics.db` persists
  only through the dataset sync.
