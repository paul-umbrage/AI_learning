# Deploy frontend + backend on Vercel (with Neon Postgres, no Redis)

This project is set up to run **entirely on Vercel**: Angular frontend and FastAPI backend as a serverless function. Postgres with pgvector is provided by **Neon** (free tier). Redis is **disabled** for this setup.

---

## What’s configured

- **Root `vercel.json`** – Builds the frontend, serves it, and routes `/api/*` to the Python backend.
- **`api/index.py`** – Exposes the FastAPI app from `backend/main` so Vercel runs it as one serverless function.
- **`requirements.txt`** (root) – Pulls in `backend/requirements.txt` so the API has all dependencies.
- **Backend**
  - Uses **`DATABASE_URL`** for Postgres (Neon connection string).
  - **Redis** is off when **`DISABLE_REDIS=true`** (or unset in this setup).
  - **CORS** allows `*.vercel.app` and localhost.
  - **`SKIP_OPENAI_VALIDATION=true`** is optional for faster cold starts.

---

## 1. Neon: free Postgres with pgvector

1. Go to [neon.tech](https://neon.tech) and sign up.
2. Create a project and a database.
3. In the dashboard, enable the **pgvector** extension (Neon supports it; run `CREATE EXTENSION IF NOT EXISTS vector;` in the SQL editor if needed).
4. Copy the **connection string** (e.g. `postgresql://user:pass@ep-xxx.region.aws.neon.tech/neondb?sslmode=require`).

---

## 2. Vercel project and env vars

1. Push your repo to GitHub/GitLab/Bitbucket and [import it in Vercel](https://vercel.com/new).
2. Use the **root** of the repo (no “Root Directory” override).
3. Add these **Environment Variables** in the Vercel project (Settings → Environment Variables):

   | Name | Value | Notes |
   |------|--------|--------|
   | `DATABASE_URL` | Your Neon connection string | Required for Postgres + pgvector |
   | `OPENAI_API_KEY` | Your OpenAI API key | Required |
   | `DISABLE_REDIS` | `true` | No Redis on this setup |
   | `SKIP_OPENAI_VALIDATION` | `true` | Optional; faster cold starts |

4. **Optional:** set **`NG_APP_API_URL`** only if the frontend must call a different API (e.g. another domain). If you leave it unset, the frontend will use the same Vercel deployment as the API (`https://<VERCEL_URL>/api`).

---

## 3. Create tables in Neon (one-time)

Neon gives you an empty database. Run the setup SQL once:

1. Open **[Neon Console](https://console.neon.tech)** → your project → **SQL Editor** → **New query**.
2. Copy the contents of **`backend/neon_setup.sql`** and paste into the editor.
3. Click **Run**. This enables the `vector` extension and creates `pdf_chunks` and `pdf_documents`.

If the vector index step fails on an empty table, you can skip it; the app will still work. After you upload at least one PDF, you can create the index from the SQL editor or re-run the script.

---

## 4. Deploy

- Push to your connected branch; Vercel will build and deploy.
- **Build:** runs `cd frontend && npm ci && npm run build:vercel` and uses `frontend/dist/frontend/browser` as output.
- **API:** `/api/*` is handled by the Python function (e.g. `/api/health`, `/api/chat`, `/api/upload`, `/api/pdfs`).

---

## 5. Limits to be aware of

- **Function timeout:** 60s on Hobby (configurable in `vercel.json`). Large PDF uploads might approach this.
- **Bundle size:** Python serverless bundle is capped at **250 MB (unzipped)**. This app’s backend (FastAPI + numpy, pdfplumber, pillow, psycopg2, etc.) can exceed that, so the **API may fail to deploy** with “Serverless Function has exceeded the unzipped maximum size of 250 MB”.
- **If you hit the 250 MB limit:** Deploy **only the frontend** on Vercel and run the **backend** on Railway, Render, or Fly.io (see `VERCEL_DEPLOYMENT.md`). Set `NG_APP_API_URL` in Vercel to your backend URL.
- **Cold starts:** First request after idle can be slow; `SKIP_OPENAI_VALIDATION=true` reduces startup work.

---

## Summary checklist

- [ ] Neon project created; pgvector enabled; `DATABASE_URL` copied.
- [ ] Vercel project created from repo root; env vars set (`DATABASE_URL`, `OPENAI_API_KEY`, `DISABLE_REDIS`, optionally `SKIP_OPENAI_VALIDATION`).
- [ ] Tables created in Neon (run `create_tables()` once).
- [ ] Deploy; open the Vercel URL and test chat and PDF upload.
