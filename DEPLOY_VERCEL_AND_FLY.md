# Deploy: Frontend on Vercel, Backend on Fly.io

Complete app setup: **Vercel** for the Angular frontend, **Fly.io** for the FastAPI backend, **Neon** for Postgres.

---

## Order of operations

1. **Backend on Fly.io** (so you have an API URL).
2. **Frontend on Vercel** (connects to that API).

---

## 1. Deploy backend on Fly.io

Follow **[FLY_DEPLOYMENT.md](./FLY_DEPLOYMENT.md)**:

```bash
cd backend
fly launch --no-deploy
fly secrets set DATABASE_URL="postgresql://..." OPENAI_API_KEY="sk-..." DISABLE_REDIS="true"
fly deploy
```

Note your backend URL, e.g. **https://ai-learning-api.fly.dev**  
API base for the frontend: **https://ai-learning-api.fly.dev/api**

---

## 2. Deploy frontend on Vercel

**Note:** The repo has an `api.disabled/` folder (renamed from `api/`) so Vercel only builds the frontend and does not try to deploy a Python serverless function (which would exceed size limits). Backend runs on Fly.io only.

1. Push your repo and go to **[vercel.com/new](https://vercel.com/new)**. Import the repo.
2. **Root Directory:** leave as **empty** (use repo root). The root `vercel.json` defines the frontend build.
3. **Build & Output (from vercel.json):**
   - Build: `cd frontend && npm ci && npm run build:vercel`
   - Output: `frontend/dist/frontend/browser`
4. **Environment variable (optional):**  
   The app defaults to **https://ai-learning-api.fly.dev/api**.  
   If your Fly app has a different URL, add **`NG_APP_API_URL`** = `https://<your-app>.fly.dev/api` (Production, Preview, Development as needed).
5. Click **Deploy**. The frontend will be served at your Vercel URL and will call the Fly.io API.

---

## 3. You’re done

- **Frontend:** `https://<your-project>.vercel.app`
- **Backend:** `https://ai-learning-api.fly.dev` (health: `https://ai-learning-api.fly.dev/health`)

To use a different backend later, set **`NG_APP_API_URL`** in Vercel and redeploy.
