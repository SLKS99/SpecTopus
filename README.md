# SpecTopus: Automated Peak Fitting for High-Throughput Perovskite Research

## Why SpecTopus?

In the rapidly evolving field of perovskite research, self-driving laboratories are revolutionizing how we conduct experiments and analyze data. Our automated synthesis and characterization methods can produce and analyze 96 different perovskite compositions in a single run, generating photoluminescence (PL) evolution data over approximately 12 hours. This high-throughput approach results in thousands of PL spectra that need careful analysis.

### The Challenge

Each spectrum can contain multiple peaks, and these peaks can exhibit various changes:
- Peak position shifts
- Intensity variations
- Changes in Full Width at Half Maximum (FWHM)
- Multiple overlapping peaks
- Time-dependent evolution

Manually fitting and analyzing this volume of data would be:
- Time-consuming
- Error-prone
- A bottleneck in the research pipeline
- A waste of valuable researcher time that could be spent on experimental design and interpretation

### The Solution

SpecTopus is an automated peak fitting workflow that integrates seamlessly with automated synthesis and characterization methods. By leveraging an intelligent agent-based approach, it:

1. Automatically analyzes PL spectra without human intervention
2. Accurately identifies and fits multiple peaks
3. Tracks changes in peak characteristics over time
4. Provides real-time visualization of the fitting process
5. Generates comprehensive analysis reports

## Features

- **LLM-powered peak detection**: Uses Google's Gemini API to intelligently identify peaks in spectrum data
- **Real-time visualization**: Live plotting of fitting process with side-by-side comparisons
- **Interactive dashboard**: Comprehensive analysis visualization using Plotly
- **Curve fitting**: Integrates with lmfit for robust peak fitting using Gaussian and Voigt models
- **Quality metrics**: Automatic R² score tracking and fit validation
- **Batch processing**: Handles multiple wells and reads efficiently
- **Time Evolution Tracking**: Monitor changes in peak characteristics over time

## Impact

By automating the peak fitting process, SpecTopus allows researchers to:
- Focus on experimental design and interpretation
- Process large datasets quickly and accurately
- Identify trends and patterns more efficiently
- Accelerate the pace of perovskite research
- Integrate with automated synthesis workflows

This tool is an essential component in the modern self-driving laboratory, enabling truly high-throughput experimentation and analysis in perovskite research.

---

# Technical Documentation

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

### As a library (recommended)

Install locally (editable) and import:
```bash
pip install -e .
```

Example:
```python
from spectopus.tools import (
    LLMClient,
    build_agent_config,
    run_complete_analysis,
    save_all_wells_results,
    export_peak_data_to_csv,
)

# Configure LLM (Gemini) with optional throttle
llm = LLMClient(provider="gemini", api_key="YOUR_KEY", min_delay_seconds=1.0)

# Build data config
cfg = build_agent_config(
    data_csv="your_data.csv",
    composition_csv="your_composition.csv",
    read_selection="all",
    start_wavelength=420,
    end_wavelength=860,
    wavelength_step_size=2,
)

# Analyze a well
result = run_complete_analysis(
    cfg,
    well_name="A1",
    llm=llm,
    reads="auto",
    max_peaks=4,
    r2_target=0.90,
    max_attempts=3,
    save_plots=True,
)

# Save consolidated results
all_results = [result] if not isinstance(result, list) else result
save_all_wells_results(all_results, "results/all_wells_comprehensive_analysis.json")
export_peak_data_to_csv(all_results, "results/peak_data_export.csv")
```

### Run the demo script

```bash
python fitting_agent_demo.py
```

Key configuration in `fitting_agent_demo.py`:
- `DATA_FILE`, `COMPOSITION_FILE`
- `READS_TO_ANALYZE`:
  - int (e.g., `59`)
  - list of ints (e.g., `[1, 2, 3]`)
  - string "auto" / "all"
  - comma/range string (e.g., `"2"` or `"1,3-5"`)
- `WAVELENGTH_START/END`, `MAX_PEAKS`, `R2_TARGET`, `MAX_ATTEMPTS`
- `SAVE_PNG_PLOTS`, `EXPORT_CSV`
- `wells_to_ignore`: list or comma string (e.g., `"A1,B3"`) to drop wells

### Outputs
- `analysis_output/`: PNG plots
- `results/`:
  - `all_wells_comprehensive_analysis.json`
  - `analysis_summary.txt`
  - `fit_results_<well>.png` per well
  - `peak_data_export.csv`

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