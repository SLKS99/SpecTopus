# Spectropus LLM Fitting Agent

A Python-based system for analyzing photoluminescence spectra using Large Language Models (LLMs) and traditional curve fitting techniques.

## Features

- **LLM-powered peak detection**: Uses Google's Gemini API to intelligently identify peaks in spectrum data
- **Multiple analysis methods**: Supports both numeric data analysis and image-based peak detection
- **Curve fitting**: Integrates with lmfit for robust peak fitting using Gaussian and Voigt models
- **Data processing**: Handles CSV data files with multiple reads and wells
- **Visualization**: Generates plots showing original data, detected peaks, and fitting results

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

Set your Google Gemini API key as an environment variable:

```bash
# Windows PowerShell:
$env:GOOGLE_API_KEY="your_api_key_here"

# Windows Command Prompt:
set GOOGLE_API_KEY=your_api_key_here

# macOS/Linux:
export GOOGLE_API_KEY="your_api_key_here"
```

Or modify the demo files to include your API key directly.

## Usage

### Simple Demo

Run the basic demonstration with synthetic data:

```bash
python simple_demo.py
```

This will:
- Create synthetic photoluminescence spectrum data
- Analyze it using the LLM
- Apply traditional peak detection algorithms
- Generate visualization plots

### Advanced Demo

For more complex analysis with real CSV data:

```bash
python fitting_agent_demo.py
```

## File Structure

```
SpecTopus/
├── tools/
│   ├── fitting_agent.py    # Main analysis engine
│   └── instruct.py         # LLM prompts and instructions
├── simple_demo.py          # Basic demonstration
├── fitting_agent_demo.py   # Advanced demonstration
├── sample_data.csv         # Example spectrum data
├── sample_composition.csv  # Example composition data
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## API Key Setup

You need a Google Gemini API key to use the LLM features:

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Set it as an environment variable or include it in your scripts

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Plotting and visualization
- `google-generativeai`: Google Gemini LLM API
- `lmfit`: Curve fitting library
- `Pillow`: Image processing
- `scipy`: Scientific computing (optional, for advanced peak detection)

## Output Files

The demos generate several output files:

- `synthetic_spectrum.png`: Plot of the input spectrum data
- `peak_detection.png`: Results showing detected peaks
- `analysis_A1.png`: Detailed analysis plots (when using real data)

## Customization

### Adding Your Own Data

1. Prepare your CSV files in the expected format (see `sample_data.csv` for reference)
2. Update the configuration in the demo scripts
3. Modify the file paths and parameters as needed

### Modifying Analysis Parameters

You can adjust various parameters in the demo scripts:

- `max_peaks`: Maximum number of peaks to detect
- `r2_target`: Target R² value for curve fitting
- `model_kind`: Fitting model ("gaussian" or "voigt")
- `wavelength_step_size`: Resolution of wavelength data

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed and the virtual environment is activated
2. **API key errors**: Verify your Gemini API key is correctly set
3. **CSV parsing errors**: Check that your data files match the expected format
4. **Plot generation issues**: Ensure matplotlib backend is properly configured

### Getting Help

If you encounter issues:

1. Check that all dependencies are installed: `pip list`
2. Verify your API key is working: Test with a simple LLM call
3. Check file paths and permissions
4. Review the error messages for specific guidance

## License

See LICENSE file for details.
