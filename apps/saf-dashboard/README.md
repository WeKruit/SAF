# SAF X-13 Research Dashboard

Private, evidence-first workspace for the registered NFL 2025 X-13 study. The
interface replays canonical game state, loads one logical market at a time, and
queries bounded source-time association previews without making latency,
causality, execution, or tradable-alpha claims.

## Runtime data flow

1. The browser requests `/api/dashboard/catalog`.
2. Selecting a game requests only that game's content-addressed core asset.
3. The contract inventory and association preview load only on explicit
   researcher action.
4. A market-series asset loads only after a logical market is selected.
5. The server-side gateway verifies the latest pointer, publish manifest,
   byte length, and SHA-256 before returning private JSON.

S3 credentials are server-only runtime values:

```text
PM_DATA_S3_ENDPOINT
PM_DATA_S3_REGION
PM_DATA_S3_BUCKET
PM_DATA_S3_ACCESS_KEY_ID
PM_DATA_S3_SECRET_ACCESS_KEY
PM_DATA_S3_PREFIX
```

Do not expose these values through browser props, client environment variables,
or committed files.

## Commands

```bash
npm install
npm run dev
npm test
npm run lint
npm run build
```

`npm test` builds the complete Vinext application and runs client, gateway, and
server-rendered workspace tests. Deployment is managed by the existing Sites
project in `.openai/hosting.json`.
