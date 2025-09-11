"""
Demo script for analyzing PL spectrum data using the fitting agent.
This script demonstrates the complete workflow by calling functions from fitting_agent.py.
"""

import os
from tools.fitting_agent import (
    LLMClient, 
    build_agent_config,
    curate_dataset,
    run_complete_analysis
)


def main():
    print("=== PL Spectrum Analysis Demo ===")
    
    # Set up the API key
    os.environ["GOOGLE_API_KEY"] = "AIzaSyB-3zT32fNofbvF7_WbR1UfY0RCm2QglZw"
    
    # Initialize LLM client
    print("Initializing LLM client...")
    llm = LLMClient()
    
    # Configure data processing
    print("Setting up data configuration...")
    config = build_agent_config(
        data_csv="PEASnPbI4-Toluene.csv",           # Your PL data file
        composition_csv="2D-3D (1).csv",            # Your composition file
        read_selection="1",                         # Use read 1
        wells_to_ignore=None,                       # Don't ignore any wells
        start_wavelength=500,                       # Emission start 500nm
        end_wavelength=850,                         # Emission stop 850nm
        wavelength_step_size=1,                     # Step 1nm
        fill_na_value=0.0
    )
    
    print("Loading and curating dataset...")
    try:
        # Load and process the data
        curated = curate_dataset(config)
        print("Available wells:", curated["wells"][:10], "...")
        print("Available reads:", curated["reads"])
        
        # Analyze all wells for read 1
        available_wells = curated["wells"]
        print(f"\nFound {len(available_wells)} wells to analyze")
        
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
                    read=1,
                    max_peaks=3,
                    model_kind="gaussian",
                    r2_target=0.90,
                    max_attempts=3
                )
                all_results.append(results)
                
                # Quick summary for each well
                fit_result = results['fit_result']
                print(f"✅ {well_name}: {len(results['llm_numeric_result'].peaks)} peaks, R²={fit_result.stats.r2:.3f}")
                
            except Exception as e:
                print(f"❌ {well_name}: Error - {e}")
                continue
        
        # Display summary results
        print(f"\n=== Analysis Summary ===")
        print(f"Successfully analyzed {len(all_results)} out of {len(available_wells)} wells")
        
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
        
        print(f"\n=== Demo completed! ===")
        print(f"Analyzed {len(all_results)} wells. Check the generated files for detailed results.")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        print("Make sure your CSV files are in the correct format and accessible.")


if __name__ == "__main__":
    main()