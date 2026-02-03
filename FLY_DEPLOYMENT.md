# Deploy backend on Fly.io

Use this when the **frontend** is on Vercel and the **backend** runs on Fly.io. The backend uses your existing **Neon** Postgres (no Fly Postgres required).

---

## Prerequisites

- [Fly.io CLI](https://fly.io/docs/hands-on/install-flyctl/) installed (`brew install flyctl` or see link)
- Logged in: `fly auth login`
- **Neon** database already set up (connection string from [Neon Console](https://console.neon.tech))
- **OpenAI** API key

---

## 1. Create the app (first time only)

From the **backend** directory:

```bash
cd backend
fly launch --no-deploy
```

- When prompted for app name, use e.g. `ai-learning-api` (or accept the generated one).
- Choose a region near you.
- Do **not** add a Postgres database (we use Neon).
- This creates/updates `fly.toml` and registers the app.

---

## 2. Set secrets (env vars)

Set your Neon URL and OpenAI key (and optionally disable Redis):

```bash
fly secrets set DATABASE_URL="postgresql://USER:PASSWORD@HOST/DB?sslmode=require"
fly secrets set OPENAI_API_KEY="sk-..."
fly secrets set DISABLE_REDIS="true"
```

Use your real **Neon** connection string. To add more later: `fly secrets set KEY=value`.

---

## 3. Deploy

```bash
fly deploy
```

Build and deploy will run. When it finishes, your API will be at:

**https://&lt;your-app-name&gt;.fly.dev**

So the API base URL for the frontend is: **https://&lt;your-app-name&gt;.fly.dev/api** (your routes are under `/api`).

---

## 4. Point Vercel frontend to this backend

In the **Vercel** project (frontend):

1. **Settings → Environment Variables**
2. Add: **`NG_APP_API_URL`** = `https://<your-app-name>.fly.dev/api`  
   Example: `https://ai-learning-api.fly.dev/api`
3. **Redeploy** the frontend so the new URL is baked in.

---

## 5. CORS

The backend already allows `https://*.vercel.app`. If you use a custom domain for the frontend, add it:

```bash
fly secrets set CORS_ORIGINS="https://your-domain.com"
```

---

## Useful commands

| Command | Description |
|--------|-------------|
| `fly status` | App status and URL |
| `fly logs` | Stream logs |
| `fly ssh console` | Shell into the VM |
| `fly secrets list` | List secrets (values hidden) |
| `fly deploy` | Deploy after code changes |

---

## Summary

1. `cd backend` → `fly launch --no-deploy`
2. `fly secrets set DATABASE_URL=... OPENAI_API_KEY=... DISABLE_REDIS=true`
3. `fly deploy`
4. In Vercel: set `NG_APP_API_URL` to `https://<app>.fly.dev/api` and redeploy frontend.
