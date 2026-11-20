#!/usr/bin/env bash
# Deploys the app to Azure Container Apps.
# The image gets built inside Azure Container Registry, so only the source is uploaded.
#
#   az login
#   bash scripts/deploy_azure.sh
#
# Running it again rebuilds the image and updates the app.
set -euo pipefail

LOCATION="${LOCATION:-southafricanorth}"
RG="${RG:-pumpsense-rg}"
APP="${APP:-pumpsense}"
ENV_NAME="${ENV_NAME:-pumpsense-env}"
SUB_ID=$(az account show --query id -o tsv)
ACR="${ACR:-pumpsense${SUB_ID:0:8}}"   # registry names have to be unique across all of Azure
ACR="${ACR//-/}"
TAG="v$(date +%Y%m%d%H%M%S)"

cd "$(dirname "$0")/.."
echo "subscription: $(az account show --query name -o tsv), region: $LOCATION"

az extension add --name containerapp --upgrade --only-show-errors
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry; do
  az provider register --namespace "$ns" --wait --only-show-errors
done

az group create -n "$RG" -l "$LOCATION" -o none
az acr show -n "$ACR" -o none 2>/dev/null \
  || az acr create -n "$ACR" -g "$RG" -l "$LOCATION" --sku Basic --admin-enabled true -o none

# .dockerignore keeps the raw dataset out of the upload
az acr build -r "$ACR" -t "pumpsense:$TAG" -t pumpsense:latest .

IMAGE="$ACR.azurecr.io/pumpsense:$TAG"
ACR_USER=$(az acr credential show -n "$ACR" --query username -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR" --query "passwords[0].value" -o tsv)

az containerapp env show -n "$ENV_NAME" -g "$RG" -o none 2>/dev/null \
  || az containerapp env create -n "$ENV_NAME" -g "$RG" -l "$LOCATION" -o none

if az containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null; then
  az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" -o none
else
  # 1 replica only: the sqlite db and retrained models live inside the container.
  # 2 GB because retraining needs ~1.9 GB
  az containerapp create -n "$APP" -g "$RG" --environment "$ENV_NAME" --image "$IMAGE" \
    --registry-server "$ACR.azurecr.io" --registry-username "$ACR_USER" --registry-password "$ACR_PASS" \
    --target-port 7860 --ingress external --cpu 1 --memory 2Gi \
    --min-replicas 1 --max-replicas 1 -o none
fi

echo "deployed: https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
