#!/usr/bin/env bash
# One-shot deploy of PolicyLens to Google Cloud Run.
# Usage: ./deploy.sh PROJECT_ID NVIDIA_NIM_API_KEY
set -euo pipefail

PROJECT_ID="${1:?Usage: ./deploy.sh PROJECT_ID NIM_API_KEY}"
API_KEY="${2:?Usage: ./deploy.sh PROJECT_ID NIM_API_KEY}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-policylens}"
SECRET_NAME="${SECRET_NAME:-nim-api-key}"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com

# Store / update the NVIDIA key in Secret Manager.
if gcloud secrets describe "$SECRET_NAME" >/dev/null 2>&1; then
  echo -n "$API_KEY" | gcloud secrets versions add "$SECRET_NAME" --data-file=-
else
  echo -n "$API_KEY" | gcloud secrets create "$SECRET_NAME" --data-file=- --replication-policy=automatic
fi

# Let the Cloud Run runtime service account read the secret.
PROJECT_NUM=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding "$SECRET_NAME" \
  --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" --quiet

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --set-secrets "NIM_API_KEY=${SECRET_NAME}:latest" \
  --memory 512Mi \
  --min-instances 1 \
  --max-instances 3 \
  --cpu-boost \
  --timeout 60 \
  --quiet

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')
echo "✅ Deployed: $URL"
