/**
 * Writes environment.prod.ts with API URL from NG_APP_API_URL (e.g. set in Vercel).
 * Run before `ng build` so production build uses the correct backend URL.
 */
const fs = require('fs');
const path = require('path');

// Same-origin when frontend+backend on Vercel; override with NG_APP_API_URL if needed
const vercelUrl = process.env.VERCEL_URL;
const raw = process.env.NG_APP_API_URL && process.env.NG_APP_API_URL.trim();
const apiUrl =
  (raw && raw.length > 0 ? raw : null) ||
  (vercelUrl ? `https://${vercelUrl}/api` : null) ||
  'https://ai-learning-api.fly.dev/api';
const outPath = path.join(__dirname, '../src/environments/environment.prod.ts');
const content = `export const environment = {
  production: true,
  apiUrl: '${apiUrl.replace(/'/g, "\\'")}',
};
`;

fs.writeFileSync(outPath, content, 'utf8');
console.log('Wrote environment.prod.ts with apiUrl:', apiUrl);
