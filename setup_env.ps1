# Setup script for SpecTopus environment
Write-Host "Setting up SpecTopus environment..." -ForegroundColor Green

# Activate virtual environment
& "venv\Scripts\Activate.ps1"



# TODO: Set your Google API key before running (do not commit real keys)
$env:GOOGLE_API_KEY=""
# Throttle Gemini calls to reduce RPD spikes (1 second between calls)
$env:GEMINI_MIN_DELAY_MS="1000"


# Verify setup
Write-Host "✅ Virtual environment activated" -ForegroundColor Green
Write-Host "✅ API key set" -ForegroundColor Green
Write-Host ""
Write-Host "Ready to run: python fitting_agent_demo.py" -ForegroundColor Yellow