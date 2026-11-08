# Azure deployment

Status: local implementation and verification in progress. A public URL is recorded only after successful deployment and external smoke tests.

## Hosting configuration

The default image serves React and FastAPI from one Azure Container App. This keeps one public origin and reduces deployment dependencies. Azure Static Web Apps is an optional frontend alternative.

- Container Apps Consumption workload profile, CPU inference only.
- Start with 0.5 vCPU and 1 GiB RAM; validate observed memory.
- Minimum replicas 0; maximum replicas 1; one Uvicorn worker.
- Private Azure Container Registry Basic; admin credentials disabled.
- User-assigned managed identity with AcrPull on the registry.
- No GPU, Azure ML cluster, LLM, vector database or RAG infrastructure.
- No Log Analytics workspace by default; inspect container logs on demand.

Scale-to-zero reduces compute use but can delay the first request. Registry storage, networking and other services may still incur charges. Student credit is not lifetime-free hosting. Budget alerts do not stop spending; resource, replica and request limits are separate controls.

## First deployment

After Azure CLI authentication, run `infra/deploy.ps1` from the repository root with your subscription, registry name and allowed region. It builds source locally, pushes the image and provisions the app. Never commit credentials, access tokens or deployment secrets. Verify permitted regions and available credit first.

```powershell
./infra/deploy.ps1 -SubscriptionId '<subscription>' -Registry '<unique-registry>' -Location '<allowed-region>'
```

The image includes versioned artifacts and frontend. Retraining is an explicit release operation; it never happens on startup.

## Automated delivery

`ci.yml` verifies Python tests, UI build, browser flows and Docker build. `deploy.yml` requires successful verification of the same commit, uses Azure OIDC and publishes a revision. Production-environment settings:

| Kind | Name |
|---|---|
| Secret | AZURE_CLIENT_ID |
| Secret | AZURE_TENANT_ID |
| Secret | AZURE_SUBSCRIPTION_ID |
| Variable | AZURE_REGISTRY |
| Variable | AZURE_RESOURCE_GROUP |
| Variable | AZURE_APP_NAME |
| Variable | AZURE_ENVIRONMENT |
| Variable | AZURE_PULL_IDENTITY |

Federation must trust the exact repository/environment subject. Scope Contributor to the resource group and registry push rights to the registry. An unconfigured workflow is a recipe, not proof of operational automated delivery.

The deployment job only runs from `main`. It can create the first application after its image is pushed; subsequent runs update the revision. Provision the environment and pull identity before dispatching the workflow. Student subscriptions can restrict regions; `francecentral` is an allowed region in the verified subscription used for this project.

## Verification and rollback

Test `/health`, metadata, all six historical cases, cash calculations, UI and API docs over HTTPS. Record deployed commit/image and time. Retain the previous image tag; roll back by updating the app to it and repeating smoke tests. Keep artifact, code and schema versions together.

Official references: [billing](https://learn.microsoft.com/azure/container-apps/billing), [scaling](https://learn.microsoft.com/azure/container-apps/scale-app), [managed identity image pull](https://learn.microsoft.com/azure/container-apps/managed-identity-image-pull), [GitHub Actions](https://learn.microsoft.com/azure/container-apps/github-actions).
