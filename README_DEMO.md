#  PL Spectrum Analysis Demo - Quick Start Guide

##  How to Run

### 1. Setup Environment
```powershell
# Activate virtual environment and set API key
.\setup_env.ps1
```

### 2. Run Analysis
```powershell
python fitting_agent_demo.py
```

##  What You Can Change

### In `fitting_agent_demo.py` - Configuration Section:

| Parameter | Description | Examples |
|-----------|-------------|----------|
| `DATA_FILE` | Your PL spectrum CSV file | `"your_data.csv"` |
| `COMPOSITION_FILE` | Your composition mapping | `"your_composition.csv"` |
| `READS_TO_ANALYZE` | Which read(s) to analyze | `2`, `[1,2,3]`, `"auto"`, `"all"` |
| `WAVELENGTH_START` | Start wavelength (nm) | `400`, `500`, `600` |
| `WAVELENGTH_END` | End wavelength (nm) | `800`, `850`, `900` |
| `MAX_PEAKS` | Max peaks per spectrum | `2`, `3`, `4` |
| `R2_TARGET` | Minimum R² for good fits | `0.85`, `0.90`, `0.95` |
| `MAX_ATTEMPTS` | Max retry attempts | `2`, `3`, `5` |

### Read Options Explained:
- `2` - Analyze only read 2
- `[1,2,3]` - Analyze reads 1, 2, and 3
- `"auto"` - Use first available read
- `"all"` - Analyze all available reads

##  Outputs

### Files Created:
```
analysis_output/
├── spectrum_A1.png          # Original spectrum plots
├── spectrum_A2.png
├── fit_results_A1.png       # Fitting results with components
└── fit_results_A2.png

results/
└── all_wells_comprehensive_analysis.json  # Complete analysis data
```

### JSON Structure:
```json
{
  "wells": {
    "A1": {
      "fitting_results": {
        "quality_peaks": [           // ← USE THESE (filtered good peaks)
          {
            "center_nm": 522.77,
            "FWHM_nm": 21.95,
            "height": 21426.33
          }
        ],
        "all_peaks": [...],          // All peaks (including poor quality)
        "quality_metrics": {
          "r_squared": 0.987,
          "rmse": 466.67
        }
      }
    }
  }
}
```

##  What You Cannot Change

- **LLM Model Selection**: Automatic based on spectrum shape
- **Fitting Algorithm**: Uses `lmfit` library
- **Quality Filtering**: Uses `pick_good_peaks` function
- **Output Format**: Consolidated JSON + PNG plots

##  Troubleshooting

### Common Issues:

1. **API Key Error**
   ```
    Error initializing LLM client
   ```
   **Fix**: Run `.\setup_env.ps1` to set GOOGLE_API_KEY

2. **File Not Found**
   ```
   Error processing data: file not found
   ```
   **Fix**: Put your CSV files in the main directory

3. **Poor Fits**
   ```
   Well A1: 0 quality peaks found
   ```
   **Fix**: Lower `R2_TARGET` or increase `MAX_ATTEMPTS`

### Getting Help:
- Check the PNG plots in `analysis_output/` to see what's happening
- Look at `quality_metrics` in the JSON for R² values
- Compare `all_peaks` vs `quality_peaks` to see filtering

##  Understanding Results

### Good Results:
- **R² > 0.90**: Excellent fit
- **Multiple quality_peaks**: Well-resolved peaks
- **FWHM 10-100 nm**: Reasonable peak widths

### Poor Results:
- **R² < 0.85**: Poor fit, may need more attempts
- **Empty quality_peaks**: Peaks didn't pass quality filters
- **FWHM > 200 nm**: Very broad peaks, check data quality

---
** Tip**: Focus on the `quality_peaks` in the JSON output - these are your final, filtered results!
