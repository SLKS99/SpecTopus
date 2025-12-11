"""
=== PL Spectrum Analysis Demo ===

This is a SIMPLE DEMO script that analyzes photoluminescence spectra.
All the complex functions are defined in tools/fitting_agent.py and tools/instruct.py.
This demo just configures parameters and runs the analysis.

 WHAT YOU CAN CHANGE (Configuration Section Below):
   - DATA_FILE: Your CSV data file name
   - COMPOSITION_FILE: Your composition mapping file  
   - READS_TO_ANALYZE: Which read(s) to analyze
   - WAVELENGTH_RANGE: Start and end wavelengths
   - MAX_PEAKS: Maximum peaks to find per spectrum
   - R2_TARGET: Minimum R² for good fits (0.90 = 90%)
   - MAX_ATTEMPTS: How many fitting attempts before giving up

 WHAT YOU CANNOT CHANGE (Fixed Logic):
   - LLM model selection (automatic based on spectrum shape)
   - Fitting algorithms (uses lmfit library)
   - Quality filtering (uses pick_good_peaks function)
   - Output format (consolidated JSON + PNG plots)

 HOW TO RUN:
   1. Set GOOGLE_API_KEY environment variable (use setup_env.ps1)
   2. Activate virtual environment: .\\venv\\Scripts\\Activate.ps1
   3. Run: python fitting_agent_demo.py

OUTPUTS:
   - analysis_output/: All PNG plots and images
   - results/: Consolidated JSON with all analysis results
"""

import os
from pathlib import Path
from tools.fitting_agent import (
    LLMClient, 
    build_agent_config,
    curate_dataset,
    run_complete_analysis,
    save_all_wells_results,
    export_peak_data_from_json
)

def find_file(filename, required=True):
    """Find a file by checking multiple possible locations."""
    # Get the script's directory
    script_dir = Path(__file__).parent.absolute()
    
    # Possible locations to check
    locations = [
        script_dir / filename,  # Same directory as script
        script_dir.parent / filename,  # Parent directory
        Path(filename),  # Absolute path or current working directory
        Path.cwd() / filename,  # Current working directory
    ]
    
    for loc in locations:
        if loc.exists():
            print(f"Found file: {loc}")
            return str(loc)
    
    # File not found
    if required:
        print(f"Error: Could not find file '{filename}' in any of these locations:")
        for loc in locations:
            print(f"  - {loc}")
        raise FileNotFoundError(f"Could not find file: {filename}")
    else:
        print(f"Warning: Could not find file '{filename}' - it may not be required")
        return None

def main():
    print("=== PL Spectrum Analysis Demo ===")
    
    # ========================================
    #  CONFIGURATION SECTION (CHANGE THESE)
    # ========================================
    
    # Data Files (put your files in the main directory)
    DATA_FILE = find_file("12-3-25 FA 5-aa- bda time tern 0.3 m 9 to 1.csv")  # Your PL spectrum data
    COMPOSITION_FILE = find_file("1-27-22 - MAPbI3, FAPbI3 and CsPbI3 - Compositions.csv", required=False)  # Your composition mapping (optional)
    
    # If composition file not found, try to continue without it or set a default
    if COMPOSITION_FILE is None:
        print("Warning: Composition file not found. Analysis will continue without composition mapping.")
        COMPOSITION_FILE = "2D-3D (1).csv"  # Will be handled by the fitting_agent if it can work without it
    
    # Read Selection Options:
    READS_TO_ANALYZE = 59       # Single read: 1
    # READS_TO_ANALYZE = [1,2,3]  # Multiple reads: [1,2,3]  
    # READS_TO_ANALYZE = "auto"   # Auto: first available
    # READS_TO_ANALYZE = "all"    # All available reads
    
    # Wavelength Range (nm)
    WAVELENGTH_START = 420      # Start wavelength
    WAVELENGTH_END = 860        # End wavelength
    WAVELENGTH_STEP = 2         # Step size
    
    # Peak Fitting Parameters
    MAX_PEAKS = 3               # Maximum peaks to find per spectrum
    R2_TARGET = 0.90            # Minimum R² for good fits (0.90 = 90%)
    MAX_ATTEMPTS = 3            # Max retry attempts for poor fits
    
    # Output Options
    SAVE_PNG_PLOTS = True       # Set to False to skip saving final PNG files (keeps LLM analysis)
    EXPORT_CSV = True           # Set to True to export peak data as CSV file
    
    # ========================================
    #  ANALYSIS EXECUTION (DON'T CHANGE)
    # ========================================
    
    # Initialize Gemini LLM client
    print("Initializing Gemini LLM client...")
    
    # API Key Configuration
    # IMPORTANT: Provide your API key via environment (recommended).
    env_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
    if not env_key:
        raise ValueError("No API key found. Please set GOOGLE_API_KEY or GEMINI_API_KEY.")
    api_key = env_key
    
    # CRITICAL: Always set the API key in the environment BEFORE creating LLMClient
    os.environ['GOOGLE_API_KEY'] = api_key
    os.environ['GEMINI_API_KEY'] = api_key  # Set both for compatibility
    
    print(f"[OK] API key configured (length: {len(api_key)})")
    print(f"[OK] Environment variables set: GOOGLE_API_KEY={bool(os.environ.get('GOOGLE_API_KEY'))}, GEMINI_API_KEY={bool(os.environ.get('GEMINI_API_KEY'))}")
    
    # Initialize LLM client with API key
    # Model: gemini-2.5-flash-lite (as specified)
    print(f"\n{'='*60}")
    print("Initializing LLM Client with API Key")
    print(f"{'='*60}")
    print(f"API Key (first 20 chars): {api_key[:20]}...")
    print(f"API Key length: {len(api_key)}")
    print(f"Model ID: gemini-2.5-flash-lite")
    print(f"Environment GOOGLE_API_KEY set: {'GOOGLE_API_KEY' in os.environ}")
    print(f"Environment GEMINI_API_KEY set: {'GEMINI_API_KEY' in os.environ}")
    
    try:
        llm = LLMClient(provider="gemini", model_id="gemini-2.5-flash-lite", api_key=api_key)
        print("[OK] LLM client initialized successfully")
        
        # Verify the LLMClient has the API key stored
        if hasattr(llm, 'api_key'):
            print(f"[OK] LLMClient has API key stored: {bool(llm.api_key)}")
        if hasattr(llm, 'model_id'):
            print(f"[OK] LLMClient model ID: {llm.model_id}")
        
        # Test the API key with a simple call (using the correct method name: generate, not generate_text)
        print("\nTesting API key with a simple call...")
        try:
            test_response = llm.generate("Say 'test' if you can read this.", max_tokens=10)
            print(f"[OK] API key test successful! Response: {test_response[:50]}")
            print(f"{'='*60}\n")
        except Exception as test_error:
            error_msg = str(test_error)
            print(f"\n[ERROR] API key test FAILED")
            print(f"Error: {error_msg}")
            if "API_KEY_INVALID" in error_msg or "API Key not found" in error_msg:
                print("\n" + "="*60)
                print("ERROR: API KEY IS INVALID OR NOT WORKING")
                print("="*60)
                print("The API key you provided is being rejected by Google's API.")
                print("Possible reasons:")
                print("1. The API key might be invalid or expired")
                print("2. The Generative AI API might not be enabled for this key")
                print("3. The API key might not have the correct permissions")
                print("4. The model 'gemini-2.5-flash-lite' might not be available for this API key")
                print("\nTo fix this:")
                print("1. Go to https://makersuite.google.com/app/apikey")
                print("2. Create a new API key or verify your existing one")
                print("3. Make sure the Generative AI API is enabled")
                print("4. Try a different model if gemini-2.5-flash-lite doesn't work")
                print("5. Update the API key in this script or set it as an environment variable")
                print("="*60 + "\n")
                raise ValueError(f"Invalid API key: {test_error}")
            else:
                print(f"Warning: API test failed with error: {test_error}")
                print("Continuing anyway, but you may encounter errors...")
    except Exception as e:
        print(f"\n[ERROR] Error initializing LLM client: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    # Configure data processing
    print("Setting up data configuration...")
    # Handle read_selection properly - can be int, list, or string
    # Note: build_agent_config expects a list for single read, not an int
    if isinstance(READS_TO_ANALYZE, (list, tuple)):
        # For multiple reads, pass as list
        read_selection = READS_TO_ANALYZE
    elif isinstance(READS_TO_ANALYZE, int):
        # For single read, convert to list to avoid iteration errors
        read_selection = [READS_TO_ANALYZE]
    else:
        # For "auto" or "all", pass as string
        read_selection = READS_TO_ANALYZE
    
    config = build_agent_config(
        data_csv=DATA_FILE,
        composition_csv=COMPOSITION_FILE,
        read_selection=read_selection,
        wells_to_ignore=None,
        start_wavelength=WAVELENGTH_START,
        end_wavelength=WAVELENGTH_END,
        wavelength_step_size=WAVELENGTH_STEP,
        fill_na_value=0.0
    )
    
    print("Loading and curating dataset...")
    try:
        # Load and process the data
        curated = curate_dataset(config)
        print("Available wells:", curated["wells"][:10], "...")
        print("Available reads:", curated["reads"])
        
        # Analyze all wells with flexible read selection
        available_wells = curated["wells"]
        print(f"\nFound {len(available_wells)} wells to analyze")
        print("Read options: single int (2), list ([1,2,3]), 'auto' (first available), or 'all' (all available)")
        
        # Verify API key is still working before starting analysis
        print("\n=== Verifying API Key Before Analysis ===")
        try:
            # Quick test to ensure API key is still valid
            verify_response = llm.generate("Test", max_tokens=5)
            print("[OK] API key verification successful - ready to start analysis")
        except Exception as verify_error:
            error_msg = str(verify_error)
            if "API_KEY_INVALID" in error_msg or "API Key not found" in error_msg:
                print(f"\n[ERROR] API KEY VERIFICATION FAILED: {error_msg}")
                print("\nThe API key is not working. Please check:")
                print("1. The API key is correct")
                print("2. The API key has access to gemini-2.5-flash-lite model")
                print("3. The API key hasn't been revoked or expired")
                raise ValueError(f"API key verification failed: {verify_error}")
            else:
                print(f"⚠ API key verification warning: {verify_error}")
                print("Continuing with analysis, but errors may occur...")
        
        # Run complete analysis for all wells
        print("\n=== Running Complete Analysis for All Wells ===")
        all_results = []
        
        for i, well_name in enumerate(available_wells):
            print(f"\n--- Analyzing Well {well_name} ({i+1}/{len(available_wells)}) ---")
            
            try:
                results = run_complete_analysis(
                    config=config,
                    well_name=well_name,
                    llm=llm,
                    reads=READS_TO_ANALYZE,
                    max_peaks=MAX_PEAKS,
                    model_kind=None,  # Let LLM choose the model automatically
                    r2_target=R2_TARGET,
                    max_attempts=MAX_ATTEMPTS,
                    save_plots=SAVE_PNG_PLOTS
                )
                
                # Handle both single and multiple read results
                if isinstance(results, list):
                    # Multiple reads - add each result separately
                    all_results.extend(results)
                    # Summary for multiple reads
                    for result in results:
                        fit_result = result['fit_result']
                        print(f"{well_name} Read {result['read']}: {len(result['llm_numeric_result'].peaks)} peaks, R²={fit_result.stats.r2:.3f}, model={fit_result.model_kind}")
                else:
                    # Single read - add as single result
                    all_results.append(results)
                    # Quick summary for single read
                    fit_result = results['fit_result']
                    print(f"{well_name}: {len(results['llm_numeric_result'].peaks)} peaks, R²={fit_result.stats.r2:.3f}, model={fit_result.model_kind}")
                
            except Exception as e:
                print(f"{well_name}: Error - {e}")
                continue
        
        # Display summary results
        print(f"\n=== Analysis Summary ===")
        print(f"Successfully analyzed {len(all_results)} out of {len(available_wells)} wells")
        
        # Save overall summary
        summary_path = "results/analysis_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Successfully analyzed {len(all_results)} out of {len(available_wells)} wells\n")
            for res in all_results:
                well_name = res['well_name']
                r2 = res['fit_result'].stats.r2 if res['fit_result'].success else "N/A"
                f.write(f"{well_name}: R²={r2}\n")
        print(f" Summary saved to {summary_path}")
        
        # Show top performing wells
        successful_results = [r for r in all_results if r['fit_result'].success]
        if successful_results:
            # Sort by R²
            successful_results.sort(key=lambda x: x['fit_result'].stats.r2, reverse=True)
            
            print(f"\n=== Top 5 Best Fitting Wells ===")
            for i, results in enumerate(successful_results[:5]):
                well_name = results['well_name']
                fit_result = results['fit_result']
                print(f"{i+1}. {well_name}: R²={fit_result.stats.r2:.4f}, {len(fit_result.peaks)} peaks")
        
        # Save consolidated results
        if all_results:
            print(f"\n=== Saving Consolidated Results ===")
            consolidated_file = save_all_wells_results(all_results, "results/all_wells_comprehensive_analysis.json")
            print(f"Consolidated analysis saved to: {consolidated_file}")
            
            # Export peak data to CSV if requested
            if EXPORT_CSV:
                print(f"\n=== Exporting Peak Data to CSV ===")
                csv_file = export_peak_data_from_json(
                    consolidated_file, 
                    "results/peak_data_export.csv",
                    composition_csv=COMPOSITION_FILE
                )
                print(f"Peak data exported to: {csv_file}")
        
        print(f"\n=== Demo completed! ===")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        print("Make sure your CSV files are in the correct format and accessible.")

if __name__ == "__main__":
    main()