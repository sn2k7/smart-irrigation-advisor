# Walk through the main API endpoints against a RUNNING server (Windows PowerShell).
# Start the API first:  uvicorn irrigation_advisor.api.main:app --port 8000
# Then:                 .\scripts\demo_api.ps1
param([string]$BaseUrl = "http://127.0.0.1:8000")
$ErrorActionPreference = "Stop"
$json = "application/json"

Write-Host "Health:" -ForegroundColor Cyan
Invoke-RestMethod "$BaseUrl/health" | Format-List

Write-Host "Create a field:" -ForegroundColor Cyan
$field = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/v1/fields" -ContentType $json `
    -Body (Get-Content examples\field_create.json -Raw)
$id = $field.id
$field | Format-List

Write-Host "Upload 48 h of readings (drying soil):" -ForegroundColor Cyan
$now = (Get-Date).Date.AddHours((Get-Date).Hour)
$readings = for ($i = 47; $i -ge 0; $i--) {
    @{ field_id = $id; timestamp = $now.AddHours(-$i).ToString("yyyy-MM-ddTHH:mm:ss");
       soil_moisture = [math]::Round(19.5 - (47 - $i) * 0.04, 2); temperature_c = 29.0; humidity_pct = 55.0; rainfall_mm = 0.0 }
}
$batch = @{ readings = $readings } | ConvertTo-Json -Depth 4
$result = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/v1/readings/batch" -ContentType $json -Body $batch
"stored=$($result.stored) invalid=$($result.invalid)"

Write-Host "Recommendation (no rain forecast):" -ForegroundColor Cyan
$rec = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/v1/recommendations" -ContentType $json `
    -Body (@{ field_id = $id; forecast_rain_24h_mm = 0 } | ConvertTo-Json)
$rec.summary
$rec.reasons

Write-Host "Same field with 25 mm rain forecast:" -ForegroundColor Cyan
$rec2 = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/v1/recommendations" -ContentType $json `
    -Body (@{ field_id = $id; forecast_rain_24h_mm = 25 } | ConvertTo-Json)
$rec2.summary

Write-Host "Faulty reading (flagged, triggers fallback):" -ForegroundColor Cyan
$bad = @{ field_id = $id; timestamp = $now.AddMinutes(5).ToString("yyyy-MM-ddTHH:mm:ss"); soil_moisture = 99.9 } | ConvertTo-Json
(Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/v1/readings" -ContentType $json -Body $bad).issues | Format-Table
$rec3 = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/v1/recommendations" -ContentType $json `
    -Body (@{ field_id = $id; forecast_rain_24h_mm = 0 } | ConvertTo-Json)
"$($rec3.decision_source): $($rec3.summary)"

Write-Host "Analytics summary:" -ForegroundColor Cyan
Invoke-RestMethod "$BaseUrl/api/v1/fields/$id/analytics/summary?days=7" | Format-List

Write-Host "Model info:" -ForegroundColor Cyan
(Invoke-RestMethod "$BaseUrl/api/v1/model").message
