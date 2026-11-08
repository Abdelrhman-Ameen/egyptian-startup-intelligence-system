# Azure deployment

**Status: deployed and externally verified on 8 November 2026.**

- [Live financial workspace](https://ventureguard.agreeablesea-8b0aa1c1.francecentral.azurecontainerapps.io)
- [Live API docs](https://ventureguard.agreeablesea-8b0aa1c1.francecentral.azurecontainerapps.io/docs)
- [Successful GitHub deployment run](https://github.com/Abdelrhman-Ameen/egyptian-startup-intelligence-system/actions/runs/42444538914)
- Deployed source commit: `ce11b07269e20eb5ce2b3b5308fb5b00c1f6bd18`. Subsequent documentation/recording changes do not change this image's runtime code.
- Region: `francecentral`; HTTPS only; 0.5 vCPU / 1 GiB; replicas 0–1.
- The private registry has admin access disabled. GitHub uses OIDC and the app pulls images with a managed identity.

## Verified behavior

All 23 Python tests and four desktop/mobile browser flows passed. The four browser flows also passed against the public Azure URL. Twelve historical predictions from the live service matched frozen out-of-fold predictions within `1e-9`, and the known cash example reconciled. Sequential warm HTTP measurements from the verification client had a median of approximately 72 ms and p95 of 73 ms; this is not a load or cold-start benchmark.

Evidence: [cloud smoke checks](../reports/ventureguard/cloud_validation.json), [deployment settings](../reports/ventureguard/deployment.json), [local container checks](../reports/ventureguard/validation.json), and the [short walkthrough](https://github.com/Abdelrhman-Ameen/egyptian-startup-intelligence-system/releases/download/v1.0.0/ventureguard-demo.webm).

A project-resource-group monthly cost budget of **10 USD** is configured with actual-cost email alerts at 80% and 100%. Recipient details remain in Azure. These are notifications, not an automatic spending cap. Registry storage is billed independently of application scale-to-zero. Monitor cost and the student-credit expiry in the Azure portal.

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
