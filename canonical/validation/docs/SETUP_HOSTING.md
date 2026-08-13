# Hosting it on Google Cloud Run

## 1. Create the database

1. Sign up at https://neon.com — free tier, no card, 0.5 GB.
2. Create a project, and note which region you put it in. Step 6 picks a Google
   region to match.
3. From the connection details, note five values: database name, user, password,
   host, port (5432).

## 2. Install the Google Cloud CLI

```bash
brew install --cask google-cloud-sdk
gcloud auth login
```

## 3. Create the project and switch it on

```bash
gcloud projects create prism-review-$RANDOM --name="PRISM review"
gcloud config set project <the-project-id-it-printed>
```

Then **link a billing account** — https://console.cloud.google.com/billing —
and enable the three services used:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## 4. Set a budget alert first

Before deploying, not after. https://console.cloud.google.com/billing → **Budgets
& alerts** → create a budget of **$1** with an email alert.

You should never be charged, and `--max-instances 1` caps the runaway case, but
a tripwire costs nothing.

## 5. Write the server's settings file

```bash
cp cloudrun.yaml.example cloudrun.yaml
python3 -c "import secrets; print('ADMIN_KEY :', secrets.token_hex(16)); print('SECRET_KEY:', secrets.token_urlsafe(48))"
```

Open `cloudrun.yaml` and fill in the two generated keys, the five Neon values,
and your login. Leave `LABEL_STUDIO_HOST` empty for now — step 7 fills it in.

**`LABEL_STUDIO_USERNAME` and `LABEL_STUDIO_PASSWORD` are yours to invent.**
They are not something you fetch from anywhere: Label Studio *creates* that
account the first time it boots, and it becomes your admin login. Both must be
filled in — if either is blank no account is created, and since signup is closed
you would be locked out of your own server.

`ADMIN_KEY` must be letters and numbers only: it goes into a pattern match and
punctuation breaks it. `token_hex` is safe.

There are two settings files and they do different jobs:

| file | who reads it | holds |
| --- | --- | --- |
| `cloudrun.yaml` | the server on Cloud Run | database, passwords, gatekeeper key |
| `.env` | the scripts on your machine | the server's URL and your access token |

Both are gitignored. Keep `cloudrun.yaml` at the project root, **not** inside
`deploy/` — everything in `deploy/` is uploaded to Google's build service, and
secrets don't belong there.

## 6. Deploy

**Pick the region first.** This is a *Google Cloud* region, not the one your Neon
database is in — those are different clouds with different names. 

Set it once so it can't drift between commands:

```bash
export REGION=us-east1
```

Then deploy from the project folder — **in the same terminal**, so `$REGION` is
still set. If `--region` reports "expected one argument", `$REGION` is empty:
run the `export` line above again.

```bash
gcloud run deploy prism-review \
  --source deploy/ \
  --region $REGION \
  --env-vars-file cloudrun.yaml \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 1 \
  --timeout 600 \
  --cpu-boost \
  --startup-probe httpGet.path=/version,httpGet.port=8080,periodSeconds=5,timeoutSeconds=3,failureThreshold=40
```


## 7. Tell Label Studio its own address

The URL only exists once it has been deployed, so print it:

```bash
gcloud run services describe prism-review --region $REGION --format 'value(status.url)'
```

Put that URL into `cloudrun.yaml` **twice** — as `LABEL_STUDIO_HOST` and as
`CSRF_TRUSTED_ORIGINS`, no trailing slash on either — then run the **same deploy
command as step 6** again. It takes seconds the second time; the image is
already built.


## 8. Load the languages

Get a token: your initials (top right) → **Account & Settings** → **Personal
Access Token** → Copy. Then in `.env` on your own machine:

```
LABEL_STUDIO_URL=https://prism-review-xxxxx.run.app
LABEL_STUDIO_API_KEY=<the token>
PRISM_ADMIN_KEY=<the same ADMIN_KEY>
```

```bash
python tools/make_tasks.py --push
```

`PRISM_ADMIN_KEY` is what gets the script past the gatekeeper — the script's
equivalent of visiting `/unlock`. Without it every write returns 403.


## 9. Collect the validated work

```bash
python tools/collect.py
```

Writes `data/reviewed/full_dataset_<lang>.json` and `out/notes_<lang>.json`.


## Troubleshooting

**Build fails.** `gcloud builds log --region $REGION` or the Cloud Build
page. Usually a typo in the Dockerfile path — `--source deploy/` is required,
since that is where the Dockerfile lives.

**Container fails to start.** `gcloud run services logs read prism-review
--region $REGION --limit 50`. `start.sh` deliberately refuses to start
without `ADMIN_KEY` rather than coming up unprotected, so a missing secret shows
as an immediate exit.

**Cannot connect to the database.** Check the five `POSTGRE_*` values. Label
Studio has no SQLite fallback here — with `DJANGO_DB=default` it uses Postgres
and fails loudly, which is what you want.

**Everything is slow for everyone.** Raise `--cpu` to 4 and redeploy. Watch the
budget alert if you do.
