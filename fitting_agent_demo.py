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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import display, clear_output
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.express as px
from tools.fitting_agent import (
    LLMClient, 
    build_agent_config,
    curate_dataset,
    run_complete_analysis,
    save_all_wells_results,
    export_peak_data_from_json
)


def setup_interactive_plot():
    """Set up an interactive matplotlib plot for real-time visualization."""
    plt.ion()  # Enable interactive mode
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    
    # Top left: Raw data
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_raw.set_title('Raw Data')
    ax_raw.set_xlabel('Wavelength (nm)')
    ax_raw.set_ylabel('Intensity')
    
    # Top right: Fitted result
    ax_fit = fig.add_subplot(gs[0, 1])
    ax_fit.set_title('Fitted Result')
    ax_fit.set_xlabel('Wavelength (nm)')
    ax_fit.set_ylabel('Intensity')
    
    # Bottom: Fitting quality metrics
    ax_metrics = fig.add_subplot(gs[1, :])
    ax_metrics.set_title('Fitting Quality History')
    ax_metrics.set_xlabel('Well Number')
    ax_metrics.set_ylabel('R² Score')
    ax_metrics.set_ylim(0, 1)
    
    fig.suptitle('Real-time Peak Fitting Analysis', fontsize=14)
    plt.tight_layout()
    
    return fig, ax_raw, ax_fit, ax_metrics

def create_summary_dashboard(all_results):
    """Create an interactive plotly dashboard summarizing all analysis results."""
    # Create a subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Peak Distribution',
            'R² Score Distribution',
            'Peak Wavelengths vs Intensity',
            'Model Performance Summary'
        ),
        specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # Extract data for plotting
    r2_scores = []
    peak_counts = []
    peak_wavelengths = []
    peak_intensities = []
    model_types = {}
    
    for result in all_results:
        if isinstance(result, dict) and 'fit_result' in result:
            fit_result = result['fit_result']
            if fit_result.success:
                r2_scores.append(fit_result.stats.r2)
                peaks = fit_result.peaks
                peak_counts.append(len(peaks))
                
                # Collect peak data
                for peak in peaks:
                    peak_wavelengths.append(peak.center)
                    peak_intensities.append(peak.height)
                
                # Count model types
                model_type = fit_result.model_kind
                model_types[model_type] = model_types.get(model_type, 0) + 1
    
    # Add histograms
    fig.add_trace(
        go.Histogram(x=peak_counts, name='Peak Count Distribution'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=r2_scores, name='R² Score Distribution'),
        row=1, col=2
    )
    
    # Add scatter plot of peak wavelengths vs intensities
    fig.add_trace(
        go.Scatter(
            x=peak_wavelengths,
            y=peak_intensities,
            mode='markers',
            name='Peak Characteristics',
            marker=dict(
                size=8,
                color=peak_intensities,
                colorscale='Viridis',
                showscale=True
            )
        ),
        row=2, col=1
    )
    
    # Add bar chart of model types
    model_names = list(model_types.keys())
    model_counts = list(model_types.values())
    
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=model_counts,
            name='Model Types'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Peak Fitting Analysis Summary Dashboard",
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Number of Peaks", row=1, col=1)
    fig.update_xaxes(title_text="R² Score", row=1, col=2)
    fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
    fig.update_xaxes(title_text="Model Type", row=2, col=2)
    
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Intensity", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    # Save the dashboard
    fig.write_html("analysis_output/analysis_dashboard.html")
    
def update_plot(ax_raw, ax_fit, ax_metrics, wavelengths, intensities, fit_curve=None, peaks=None, well_name="", r2_history=None):
    """Update the interactive plot with new data."""
    # Clear all axes
    ax_raw.clear()
    ax_fit.clear()
    
    # Plot raw data
    ax_raw.plot(wavelengths, intensities, 'b-', label='Raw Data', alpha=0.8)
    ax_raw.set_title(f'Raw Data - Well {well_name}')
    ax_raw.set_xlabel('Wavelength (nm)')
    ax_raw.set_ylabel('Intensity')
    ax_raw.legend()
    ax_raw.grid(True, alpha=0.3)
    
    # Plot fitted data
    ax_fit.plot(wavelengths, intensities, 'b-', label='Raw Data', alpha=0.4)
    if fit_curve is not None:
        ax_fit.plot(wavelengths, fit_curve, 'r-', label='Fitted Curve', linewidth=2)
    
    if peaks is not None:
        peak_colors = plt.cm.rainbow(np.linspace(0, 1, len(peaks)))
        for i, (peak, color) in enumerate(zip(peaks, peak_colors)):
            # Plot peak position
            ax_fit.axvline(x=peak.center, color=color, linestyle='--', alpha=0.5)
            # Add peak label with background
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8)
            ax_fit.text(peak.center, peak.height, f'Peak {i+1}', 
                       rotation=90, verticalalignment='bottom',
                       bbox=bbox_props)
    
    ax_fit.set_title(f'Fitted Result - Well {well_name}')
    ax_fit.set_xlabel('Wavelength (nm)')
    ax_fit.set_ylabel('Intensity')
    ax_fit.legend()
    ax_fit.grid(True, alpha=0.3)
    
    # Update metrics plot
    if r2_history is not None:
        ax_metrics.clear()
        # Create color gradient based on R² values
        colors = plt.cm.RdYlGn(np.array(r2_history))
        
        # Plot points and lines
        ax_metrics.plot(range(len(r2_history)), r2_history, 'k-', alpha=0.3, zorder=1)
        ax_metrics.scatter(range(len(r2_history)), r2_history, c=colors, s=100, zorder=2)
        
        ax_metrics.set_title('Fitting Quality History')
        ax_metrics.set_xlabel('Well Number')
        ax_metrics.set_ylabel('R² Score')
        ax_metrics.set_ylim(0, 1)
        ax_metrics.grid(True, alpha=0.3)
        
        # Add target line
        ax_metrics.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target R²')
        ax_metrics.legend()
    
    plt.tight_layout()
    plt.pause(0.1)

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
        
        # Set up interactive visualization
        fig, ax_raw, ax_fit, ax_metrics = setup_interactive_plot()
        r2_history = []
        
        # Run complete analysis for all wells
        print("\n=== Running Complete Analysis for All Wells ===")
        all_results = []
        
        # Create progress bar
        for i, well_name in tqdm(enumerate(available_wells), total=len(available_wells), desc="Analyzing Wells"):
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
                        r2 = fit_result.stats.r2
                        r2_history.append(r2)
                        
                        # Update visualization
                        wavelengths = result['wavelengths']
                        intensities = result['intensities']
                        fit_curve = fit_result.best_fit if fit_result.success else None
                        update_plot(ax_raw, ax_fit, ax_metrics, wavelengths, intensities, fit_curve, 
                                  fit_result.peaks, f"{well_name} (Read {result['read']})", r2_history)
                        
                        print(f"{well_name} Read {result['read']}: {len(result['llm_numeric_result'].peaks)} peaks, R²={r2:.3f}, model={fit_result.model_kind}")
                else:
                    # Single read - add as single result
                    all_results.append(results)
                    # Quick summary for single read
                    fit_result = results['fit_result']
                    r2 = fit_result.stats.r2
                    r2_history.append(r2)
                    
                    # Update visualization
                    wavelengths = results['wavelengths']
                    intensities = results['intensities']
                    fit_curve = fit_result.best_fit if fit_result.success else None
                    update_plot(ax_raw, ax_fit, ax_metrics, wavelengths, intensities, fit_curve, 
                              fit_result.peaks, well_name, r2_history)
                    
                    print(f"{well_name}: {len(results['llm_numeric_result'].peaks)} peaks, R²={r2:.3f}, model={fit_result.model_kind}")
                
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
        
        # Save the final plot state
        plt.savefig('analysis_output/final_analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create and save interactive dashboard
        print("\nCreating interactive analysis dashboard...")
        create_summary_dashboard(all_results)
        print("Interactive dashboard saved to: analysis_output/analysis_dashboard.html")

        
    except Exception as e:
        print(f"Error processing data: {e}")
        print("Make sure your CSV files are in the correct format and accessible.")

if __name__ == "__main__":
    main()
