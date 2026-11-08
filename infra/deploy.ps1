param(
    [Parameter(Mandatory=$true)][string]$SubscriptionId,
    [string]$Location = 'switzerlandnorth',
    [string]$ResourceGroup = 'rg-ventureguard-demo',
    [Parameter(Mandatory=$true)][string]$Registry,
    [string]$AppName = 'ventureguard',
    [string]$ImageTag = 'v1.0.0'
)
$ErrorActionPreference = 'Stop'
function Invoke-Az {
    & az @args
    if ($LASTEXITCODE -ne 0) { throw "Azure command failed." }
}
Invoke-Az account set --subscription $SubscriptionId
Invoke-Az group create --name $ResourceGroup --location $Location --output none
Invoke-Az acr create --name $Registry --resource-group $ResourceGroup --sku Basic --admin-enabled false --output none
Invoke-Az acr login --name $Registry
$image = "$Registry.azurecr.io/ventureguard:$ImageTag"
docker build -t $image .
if ($LASTEXITCODE -ne 0) { throw 'Docker build failed.' }
docker push $image
if ($LASTEXITCODE -ne 0) { throw 'Docker push failed.' }
Invoke-Az identity create --name 'ventureguard-pull' --resource-group $ResourceGroup --output none
$identity = & az identity show --name 'ventureguard-pull' --resource-group $ResourceGroup | ConvertFrom-Json
$registryId = & az acr show --name $Registry --resource-group $ResourceGroup --query id -o tsv
Invoke-Az role assignment create --assignee-object-id $identity.principalId --assignee-principal-type ServicePrincipal --role AcrPull --scope $registryId --output none
Invoke-Az containerapp env create --name 'ventureguard-env' --resource-group $ResourceGroup --location $Location --logs-destination none --output none
Invoke-Az containerapp create --name $AppName --resource-group $ResourceGroup --environment 'ventureguard-env' --image $image --ingress external --target-port 8000 --transport auto --min-replicas 0 --max-replicas 1 --cpu .5 --memory 1Gi --user-assigned $identity.id --registry-identity $identity.id --registry-server "$Registry.azurecr.io" --env-vars 'OMP_NUM_THREADS=2' 'OPENBLAS_NUM_THREADS=2' --output none
Invoke-Az containerapp show --name $AppName --resource-group $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
