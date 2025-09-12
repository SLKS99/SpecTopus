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
   2. Activate virtual environment: .\venv\Scripts\Activate.ps1
   3. Run: python fitting_agent_demo.py

OUTPUTS:
   - analysis_output/: All PNG plots and images
   - results/: Consolidated JSON with all analysis results
"""

import os
from tools.fitting_agent import (
    LLMClient, 
    build_agent_config,
    curate_dataset,
    run_complete_analysis,
    save_all_wells_results,
    export_peak_data_from_json
)


def main():
    print("=== PL Spectrum Analysis Demo ===")
    
    # ========================================
    #  CONFIGURATION SECTION (CHANGE THESE)
    # ========================================
    
    # Data Files (put your files in the main directory)
    DATA_FILE = "8-29-25 Tern FAF-PEA5-AVA 9 to 1 0.4M 0.csv"  # Your PL spectrum data
    COMPOSITION_FILE = "2D-3D (1).csv"                          # Your composition mapping
    
    # Read Selection Options:
    READS_TO_ANALYZE = 2        # Single read: 2
    # READS_TO_ANALYZE = [1,2,3]  # Multiple reads: [1,2,3]  
    # READS_TO_ANALYZE = "auto"   # Auto: first available
    # READS_TO_ANALYZE = "all"    # All available reads
    
    # Wavelength Range (nm)
    WAVELENGTH_START = 500      # Start wavelength
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
    
    # Initialize Gemini LLM client (reads GOOGLE_API_KEY from environment)
    print("Initializing Gemini LLM client...")
    print(f"API key exists: {'GOOGLE_API_KEY' in os.environ}")
    

    llm = LLMClient(provider="gemini", model_id="gemini-1.5-flash")
        
 
    
    # Configure data processing
    print("Setting up data configuration...")
    config = build_agent_config(
        data_csv=DATA_FILE,
        composition_csv=COMPOSITION_FILE,
        read_selection=str(READS_TO_ANALYZE if isinstance(READS_TO_ANALYZE, int) else READS_TO_ANALYZE[0]),
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
                
                # Files are automatically organized by run_complete_analysis
                
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
        
        # Show detailed results for first well as example
        if all_results:
            example_results = all_results[0]
            well_name = example_results['well_name']
            
            print(f"\n=== Detailed Results Example (Well {well_name}) ===")
            
            # LLM Results
            print(f"\nLLM Numeric Analysis found {len(example_results['llm_numeric_result'].peaks)} peaks:")
            for i, peak in enumerate(example_results['llm_numeric_result'].peaks):
                print(f"  Peak {i+1}: center={peak.center:.1f} nm, height={peak.height:.1f}, fwhm={peak.fwhm}")
            
            print(f"\nLLM Image Analysis found {len(example_results['llm_image_result'].peaks)} peaks:")
            for i, peak in enumerate(example_results['llm_image_result'].peaks):
                print(f"  Peak {i+1}: center={peak.center:.1f} nm, height={peak.height:.1f}, fwhm={peak.fwhm}")
            
            # Fitting Results
            fit_result = example_results['fit_result']
            print(f"\n=== lmfit Fitting Results ===")
            print(f"Fitting successful: {fit_result.success}")
            print(f"R² = {fit_result.stats.r2:.4f}")
            print(f"Reduced χ² = {fit_result.stats.redchi:.2f}")
            print(f"AIC = {fit_result.stats.aic:.2f}")
            print(f"BIC = {fit_result.stats.bic:.2f}")
            
            print(f"\nPeak Parameters:")
            for i, peak in enumerate(fit_result.peaks):
                print(f"  Peak {i+1}:")
                print(f"    Position: {peak.center:.2f} nm")
                print(f"    Intensity: {peak.height:.1f}")
                print(f"    FWHM: {peak.fwhm:.2f} nm" if peak.fwhm else "    FWHM: N/A")
            
            # Quality Assessment
            print(f"\n=== Fitting Quality Assessment ===")
            for assessment_type, message in example_results['quality_assessment'].items():
                print(message)
            
            # Files Created (example)
            print(f"\n=== Files Created (Example for {well_name}) ===")
            for file_type, filename in example_results['files'].items():
                print(f"- {file_type}: {filename}")
        
        # Save consolidated results
        if all_results:
            print(f"\n=== Saving Consolidated Results ===")
            consolidated_file = save_all_wells_results(all_results, "results/all_wells_comprehensive_analysis.json")
            print(f"Consolidated analysis saved to: {consolidated_file}")
            
            # Export peak data to CSV if requested (using the clean JSON approach)
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
