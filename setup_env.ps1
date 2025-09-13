# Setup script for SpecTopus environment
Write-Host "Setting up SpecTopus environment..." -ForegroundColor Green

# Activate virtual environment
& "venv\Scripts\Activate.ps1"

# Set API key
$env:GOOGLE_API_KEY="your-api=key"

# Verify setup
Write-Host "✅ Virtual environment activated" -ForegroundColor Green
Write-Host "✅ API key set" -ForegroundColor Green
Write-Host ""
Write-Host "Ready to run: python fitting_agent_demo.py" -ForegroundColor Yellow
