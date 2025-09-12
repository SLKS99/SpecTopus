# SpecTopus Technical Documentation

A Python-based system for automated analysis of photoluminescence spectra using Large Language Models (LLMs) and advanced curve fitting techniques.

## Features

- **LLM-powered peak detection**: Uses Google's Gemini API to intelligently identify peaks in spectrum data
- **Real-time visualization**: Live plotting of fitting process with side-by-side comparisons
- **Interactive dashboard**: Comprehensive analysis visualization using Plotly
- **Curve fitting**: Integrates with lmfit for robust peak fitting using Gaussian and Voigt models
- **Quality metrics**: Automatic R² score tracking and fit validation
- **Batch processing**: Handles multiple wells and reads efficiently

## Setup

### 1. Create and Activate Virtual Environment

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API Key

Set your Google Gemini API key as an environment variable using the provided script:

```bash
# Windows PowerShell:
.\setup_env.ps1
```

Or set it manually:
```bash
# Windows PowerShell:
$env:GOOGLE_API_KEY="your_api_key_here"

# Windows Command Prompt:
set GOOGLE_API_KEY=your_api_key_here

# macOS/Linux:
export GOOGLE_API_KEY="your_api_key_here"
```

## Project Structure

```
SpecTopus/
├── tools/
│   ├── fitting_agent.py    # Main analysis engine
│   └── instruct.py         # LLM prompts and instructions
├── fitting_agent_demo.py   # Main demonstration script
├── requirements.txt        # Python dependencies
├── setup_env.ps1          # Environment setup script
└── README.md              # This documentation
```

## Usage

Run the demonstration with your PL spectrum data:

```bash
python fitting_agent_demo.py
```

### Configuration Options

You can adjust various parameters in `fitting_agent_demo.py`:

- `DATA_FILE`: Your PL spectrum data CSV file
- `COMPOSITION_FILE`: Composition mapping CSV file
- `READS_TO_ANALYZE`: Which read(s) to analyze (single, multiple, or all)
- `WAVELENGTH_START/END`: Wavelength range for analysis
- `MAX_PEAKS`: Maximum peaks to find per spectrum
- `R2_TARGET`: Minimum R² for good fits
- `MAX_ATTEMPTS`: Retry attempts for poor fits

### Output Files

The analysis generates several outputs:

- `analysis_output/`: PNG plots showing fitting results
- `analysis_output/analysis_dashboard.html`: Interactive analysis dashboard
- `results/`: 
  - `all_wells_comprehensive_analysis.json`: Complete analysis results
  - `analysis_summary.txt`: Summary statistics
  - Individual well fitting results as PNG files

## Dependencies

Core dependencies include:
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Real-time plotting
- `plotly`: Interactive dashboard
- `google-generativeai`: Gemini LLM API
- `lmfit`: Curve fitting library
- `tqdm`: Progress tracking

## Troubleshooting

### Common Issues

1. **API Key Issues**
   - Verify your Gemini API key is correctly set
   - Run `echo $env:GOOGLE_API_KEY` to check
   - Try running setup_env.ps1 again

2. **Data Format Issues**
   - Ensure your CSV files match the expected format
   - Check column names and data types
   - Verify wavelength ranges are appropriate

3. **Visualization Problems**
   - Make sure output directories exist
   - Check file permissions
   - Verify matplotlib and plotly are correctly installed

### Getting Help

If you encounter issues:
1. Check all dependencies are installed: `pip list`
2. Verify your API key is working
3. Check file paths and permissions
4. Review error messages for specific guidance

## License

See LICENSE file for details.