# fitting_agent.py
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import from instruct module
from .instruct import get_prompt

# ---------- LLM client (Gemini) ----------

# try:
#     import google.generativeai as genai  # type: ignore
# except ImportError:
#     genai = None


# class LLMClient:
#     """Lightweight wrapper for Gemini text and multimodal calls."""

#     def __init__(self, api_key: Optional[str] = None, model_id: str = "gemini-1.5-flash"):
#         key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
#         if not key:
#             raise ValueError(
#                 "No API key found. Provide api_key or set GOOGLE_API_KEY/GEMINI_API_KEY in your environment."
#             )
#         if genai is None:
#             raise ImportError("google-generativeai not installed. pip install google-generativeai")

#         genai.configure(api_key=key)
#         self.model = genai.GenerativeModel(model_id)

#     def generate(self, prompt: str, max_tokens: int = 1500) -> str:
#         """Text-only prompt. Returns plain text."""
#         try:
#             resp = self.model.generate_content(prompt, generation_config={"max_output_tokens": int(max_tokens)})
#             return getattr(resp, "text", "") or ""
#         except Exception as e:
#             logging.error(f"LLM text generation failed: {e}")
#             raise

#     def generate_multimodal(self, parts: List[Any], max_tokens: int = 1500) -> str:
#         """Multimodal prompt with [text, image, ...] parts."""
#         try:
#             resp = self.model.generate_content(parts, generation_config={"max_output_tokens": int(max_tokens)})
#             return getattr(resp, "text", "") or ""
#         except Exception as e:
#             logging.error(f"LLM multimodal generation failed: {e}")
#             raise

#. -----kamyar added ----....

try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None

try:
    from openai import OpenAI  # OpenAI official client
except ImportError:
    OpenAI = None

try:
    import anthropic  # Anthropic client
except ImportError:
    anthropic = None


class LLMClient:
    """Wrapper for multiple LLM providers (Gemini, OpenAI, Anthropic)."""

    def __init__(
        self,
        provider: str = "gemini",
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider.lower()

        if self.provider == "gemini":
            key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError("No Gemini API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
            if genai is None:
                raise ImportError("google-generativeai not installed. pip install google-generativeai")
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel(model_id or "gemini-1.5-flash")

        elif self.provider == "openai":
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("No OpenAI API key found. Set OPENAI_API_KEY.")
            if OpenAI is None:
                raise ImportError("openai not installed. pip install openai")
            self.client = OpenAI(api_key=key)
            self.model_id = model_id or "gpt-4o-mini"

        elif self.provider == "anthropic":
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("No Anthropic API key found. Set ANTHROPIC_API_KEY.")
            if anthropic is None:
                raise ImportError("anthropic not installed. pip install anthropic")
            self.client = anthropic.Anthropic(api_key=key)
            self.model_id = model_id or "claude-3-haiku-20240307"

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate(self, prompt: str, max_tokens: int = 1500) -> str:
        """Text-only generation across providers."""
        try:
            if self.provider == "gemini":
                resp = self.model.generate_content(prompt, generation_config={"max_output_tokens": int(max_tokens)})
                return getattr(resp, "text", "") or ""

            elif self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content

            elif self.provider == "anthropic":
                resp = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text if resp.content else ""

        except Exception as e:
            logging.error(f"LLM text generation failed ({self.provider}): {e}")
            raise

    def generate_multimodal(self, parts: List[Any], max_tokens: int = 1500) -> str:
        """Multimodal prompt (text+image). Currently only supported for Gemini and OpenAI."""
        try:
            if self.provider == "gemini":
                resp = self.model.generate_content(parts, generation_config={"max_output_tokens": int(max_tokens)})
                return getattr(resp, "text", "") or ""

            elif self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": str(parts[0])}] +
                            [{"type": "image_url", "image_url": {"url": p}} for p in parts[1:] if isinstance(p, str)]
                        }
                    ],
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content

            elif self.provider == "anthropic":
                raise NotImplementedError("Anthropic does not yet support multimodal input in this wrapper.")

        except Exception as e:
            logging.error(f"LLM multimodal generation failed ({self.provider}): {e}")
            raise




# ---------- Peak guess dataclasses ----------

@dataclass
class PeakGuess:
    center: float
    height: float
    fwhm: Optional[float] = None
    prominence: Optional[float] = None


@dataclass
class PeakResult:
    peaks: List[PeakGuess]
    baseline: Optional[float] = None


# ---------- Plotting (for vision path) ----------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def save_plot_png(x: np.ndarray, y: np.ndarray, outfile: str, *, title: Optional[str] = None) -> str:
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for save_plot_png")
    plt.figure()
    plt.plot(x, y)
    if title:
        plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    return outfile


def pick_good_peaks(
    out, metrics, x, y, window=(500, 850),
    min_height_snr=5.0,          # peak height ≥ SNR×RMSE
    min_area_frac=0.03,          # area fraction ≥ 3%
    fwhm_bounds=(6, 250),        # FWHM in nm
    center_margin_nm=0.5         # reject if center is pegged at bound within this margin
):
    """
    Returns: list of dicts [{id, center_nm, FWHM_nm, height, amplitude, area, frac}]
             for peaks that pass all tests.
    """
    lo, hi = window
    rmse = float(metrics.get('RMSE', np.nan))

    # parse peak ids present in metrics (p1, p2, p3, ...)
    peak_ids = sorted(
        {k.split('_')[0] for k in metrics if k.startswith('p') and k.endswith('_center')},
        key=lambda s: int(s[1:])
    )

    accepted = []
    for pid in peak_ids:
        c    = metrics.get(f'{pid}_center', np.nan)
        fwhm = metrics.get(f'{pid}_FWHM_est', np.nan)
        hgt  = metrics.get(f'{pid}_height', np.nan)
        frac = metrics.get(f'{pid}_frac', np.nan)

        # reject if parameter is pegged at its fit bounds
        pegged = False
        if out is not None and hasattr(out, "params"):
            pcenter = out.params.get(f'{pid}_center', None)
            if pcenter is not None and (np.isfinite(pcenter.min) and np.isfinite(pcenter.max)):
                if abs(pcenter.value - pcenter.min) <= center_margin_nm or abs(pcenter.value - pcenter.max) <= center_margin_nm:
                    pegged = True

        passes = (
            np.isfinite(c) and (lo <= c <= hi) and
            np.isfinite(fwhm) and (fwhm_bounds[0] <= fwhm <= fwhm_bounds[1]) and
            np.isfinite(rmse) and np.isfinite(hgt) and (hgt >= min_height_snr * rmse) and
            np.isfinite(frac) and (frac >= min_area_frac) and
            (not pegged)
        )
        if passes:
            accepted.append({
                'id': pid,
                'center_nm': float(c),
                'FWHM_nm'  : float(fwhm),
                'height'   : float(hgt),
                'amplitude': float(metrics.get(f'{pid}_amplitude', np.nan)),
                'area'     : float(metrics.get(f'{pid}_area', np.nan)),
                'frac'     : float(frac),
            })
    return accepted


def select_peak_model(model_kind: str):
    """Select appropriate lmfit model based on model_kind string."""
    if not _HAS_LMFIT:
        raise RuntimeError("lmfit is required")
    
    model_map = {
        'gaussian': GaussianModel,
        'lorentzian': LorentzianModel,
        'voigt': VoigtModel,
        'pseudovoigt': PseudoVoigtModel,
        'skewed_gaussian': SkewedGaussianModel,
        'skewed_voigt': SkewedVoigtModel,
        'exponential_gaussian': ExponentialGaussianModel,
        'split_lorentzian': SplitLorentzianModel
    }
    
    if model_kind not in model_map:
        raise ValueError(f"Unknown model_kind: {model_kind}. Available: {list(model_map.keys())}")
    
    return model_map[model_kind]


def save_fitting_plot_png(x: np.ndarray, y: np.ndarray, fit_result: PeakFitResult, outfile: str, *, title: Optional[str] = None) -> str:
    """Save a comprehensive fitting plot showing original data, fit, individual peaks, and residuals."""
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for save_fitting_plot_png")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Original data and fit
    ax1.plot(x, y, 'b-', linewidth=2, label='Original data', alpha=0.8)
    
    # Generate fit curve from peak parameters
    fit_y = np.full_like(y, fit_result.baseline or 0.0)
    colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, peak in enumerate(fit_result.peaks):
        if peak.fwhm:
            # Convert FWHM to sigma
            sigma = peak.fwhm / 2.354820045
            # Generate individual peak curve
            peak_y = peak.height * np.exp(-((x - peak.center) / sigma) ** 2 / 2)
            fit_y += peak_y
            ax1.plot(x, peak_y, '--', color=colors[i % len(colors)], 
                    alpha=0.6, linewidth=1, label=f'Peak {i+1}')
    
    ax1.plot(x, fit_y, 'r--', linewidth=2, label='Total Fit', alpha=0.9)
    
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Intensity')
    plot_title = title or f'Peak Fitting Results\nR² = {fit_result.stats.r2:.4f}, χ² = {fit_result.stats.redchi:.1f}'
    ax1.set_title(plot_title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Residuals
    residuals = y - fit_y
    ax2.plot(x, residuals, 'g-', linewidth=1, label='Residuals')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Fit Residuals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    return outfile


def save_analysis_results(analysis_result: PeakResult, well_name: str, analysis_type: str = "numeric") -> str:
    """Save LLM analysis results to JSON file."""
    analysis_data = {
        "peaks": [{"center": p.center, "height": p.height, "fwhm": p.fwhm, "prominence": p.prominence} 
                 for p in analysis_result.peaks],
        "baseline": analysis_result.baseline
    }
    filename = f'llm_analysis_{analysis_type}_{well_name}.json'
    with open(filename, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    return filename


def save_fitting_results(fit_result: PeakFitResult, well_name: str, read_num: int) -> str:
    """Save detailed fitting results to JSON file."""
    results = {
        'well': well_name,
        'read': read_num,
        'fitting_quality': {
            'success': fit_result.success,
            'r_squared': float(fit_result.stats.r2),
            'rmse': float(fit_result.stats.rmse),
            'reduced_chi_squared': float(fit_result.stats.redchi),
            'aic': float(fit_result.stats.aic),
            'bic': float(fit_result.stats.bic),
            'number_of_function_evaluations': int(fit_result.stats.nfev),
            'number_of_peaks': len(fit_result.peaks)
        },
        'peaks': [{"center": p.center, "height": p.height, "fwhm": p.fwhm, "prominence": p.prominence} 
                 for p in fit_result.peaks],
        'baseline': float(fit_result.baseline) if fit_result.baseline else None,
        'model_kind': fit_result.model_kind
    }
    filename = f'fitting_results_{well_name}_read{read_num}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    return filename


def assess_fitting_quality(fit_result: PeakFitResult) -> Dict[str, str]:
    """Assess fitting quality and return assessment messages."""
    assessments = {}
    
    # R² assessment
    if fit_result.stats.r2 > 0.95:
        assessments['r2'] = " Excellent fit (R² > 0.95)"
    elif fit_result.stats.r2 > 0.90:
        assessments['r2'] = " Good fit (R² > 0.90)"
    elif fit_result.stats.r2 > 0.80:
        assessments['r2'] = " Fair fit (R² > 0.80)"
    else:
        assessments['r2'] = " Poor fit (R² < 0.80)"
    
    # Chi-squared assessment
    if fit_result.stats.redchi < 2.0:
        assessments['chi2'] = "Good chi-squared (reduced χ² < 2.0)"
    elif fit_result.stats.redchi < 5.0:
        assessments['chi2'] = " Acceptable chi-squared (reduced χ² < 5.0)"
    else:
        assessments['chi2'] = "High chi-squared (reduced χ² > 5.0)"
    
    return assessments


def save_all_wells_results(all_results: List[Dict[str, object]], filename: str = "results/all_wells_analysis.json") -> str:
    """Save all wells analysis results to a single comprehensive JSON file."""
    consolidated_data = {
        "analysis_summary": {
            "total_wells": len(all_results),
            "successful_fits": len([r for r in all_results if r['fit_result'].success]),
            "analysis_date": pd.Timestamp.now().isoformat(),
            "model_kind": all_results[0]['fit_result'].model_kind if all_results else "unknown"
        },
        "wells": {}
    }
    
    for result in all_results:
        well_name = result['well_name']
        fit_result = result['fit_result']
        
        # Extract peak information with all details
        peaks_data = []
        for i, peak in enumerate(fit_result.peaks):
            peak_data = {
                'peak_number': i + 1,
                'position_nm': float(peak.center),
                'intensity': float(peak.height),
                'fwhm_nm': float(peak.fwhm) if peak.fwhm else None,
                'prominence': float(peak.prominence) if peak.prominence else None
            }
            peaks_data.append(peak_data)
        
        # Create comprehensive metrics for pick_good_peaks
        metrics = fit_result.best_params.copy()
        metrics['RMSE'] = fit_result.stats.rmse
        
        # Add additional metrics that pick_good_peaks expects
        total_area = sum(peak.height * peak.fwhm * np.sqrt(2 * np.pi) / 2.354820045 for peak in fit_result.peaks if peak.fwhm)
        for i, peak in enumerate(fit_result.peaks):
            prefix = f"p{i+1}"  # Use p1, p2, p3 format
            metrics[f'{prefix}_center'] = peak.center
            metrics[f'{prefix}_FWHM_est'] = peak.fwhm if peak.fwhm else np.nan
            metrics[f'{prefix}_height'] = peak.height
            peak_area = peak.height * peak.fwhm * np.sqrt(2 * np.pi) / 2.354820045 if peak.fwhm else 0
            metrics[f'{prefix}_amplitude'] = peak_area
            metrics[f'{prefix}_area'] = peak_area
            metrics[f'{prefix}_frac'] = peak_area / total_area if total_area > 0 else 0
        
        # Use pick_good_peaks to filter quality peaks
        good_peaks = pick_good_peaks(
            fit_result, metrics, result['data']['x'], result['data']['y'],
            window=(500, 850), min_height_snr=3.0, min_area_frac=0.02  # More lenient thresholds
        )
        
        consolidated_data["wells"][well_name] = {
            "read": result['read'],
            "data_info": {
                "wavelength_range": [float(result['data']['x'].min()), float(result['data']['x'].max())],
                "intensity_range": [float(result['data']['y'].min()), float(result['data']['y'].max())],
                "data_points": len(result['data']['x'])
            },
            "llm_analysis": {
                "numeric_peaks": len(result['llm_numeric_result'].peaks),
                "image_peaks": len(result['llm_image_result'].peaks),
                "numeric_baseline": result['llm_numeric_result'].baseline,
                "image_baseline": result['llm_image_result'].baseline
            },
            "fitting_results": {
                "success": fit_result.success,
                "model_kind": fit_result.model_kind,
                "quality_metrics": {
                    "r_squared": float(fit_result.stats.r2),
                    "rmse": float(fit_result.stats.rmse),
                    "reduced_chi_squared": float(fit_result.stats.redchi),
                    "aic": float(fit_result.stats.aic),
                    "bic": float(fit_result.stats.bic),
                    "function_evaluations": int(fit_result.stats.nfev)
                },
                "baseline": float(fit_result.baseline) if fit_result.baseline else None,
                "total_peaks_found": len(peaks_data),
                "quality_peaks": good_peaks,
                "all_peaks": peaks_data
            },
            "quality_assessment": result['quality_assessment'],
            "files": result['files']
        }
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(consolidated_data, f, indent=2)
    
    return filename


def assess_fit_quality_with_llm(llm: LLMClient, fit_image_path: str, r2_value: float, well_name: str) -> bool:
    """Let LLM assess fit quality from the fitting plot image."""
    fit_assessment_prompt = f"""
    Analyze this peak fitting plot for well {well_name}. The plot shows:
    - Blue line: Original spectrum data
    - Red dashed line: Total fit
    - Colored dashed lines: Individual peak components
    - Bottom panel: Residuals (difference between data and fit)
    
    Current R² = {r2_value:.4f}
    
    Assess the fitting quality by examining:
    1. How well the red fit line matches the blue data
    2. Whether individual peaks capture the actual peak shapes
    3. If residuals are randomly distributed around zero
    4. Any systematic deviations or poor peak fits
    
    Return ONLY "good" if the fit is acceptable, or "poor" if it needs improvement.
    Focus on visual fit quality, not just the R² value.
    """
    
    try:
        # Load image for multimodal analysis
        from PIL import Image
        image = Image.open(fit_image_path)
        
        # Create multimodal parts list
        parts = [fit_assessment_prompt, image]
        response = llm.generate_multimodal(parts, max_tokens=50)
        assessment = response.strip().lower()
        return "good" in assessment
    except Exception as e:
        print(f"  Error in LLM fit assessment: {e}, using R² threshold")
        return r2_value > 0.90  # Fallback to R² threshold


def select_model_for_spectrum(llm: LLMClient, x: np.ndarray, y: np.ndarray, well_name: str) -> str:
    """Let LLM select the appropriate model type for the spectrum."""
    # Create a simple spectrum summary for LLM
    spectrum_summary = f"""
    Spectrum for Well {well_name}:
    - Wavelength range: {x.min():.1f} - {x.max():.1f} nm
    - Intensity range: {y.min():.1f} - {y.max():.1f}
    - Number of peaks visible: {len(np.where(y > y.max()*0.3)[0])}
    - Peak shape: {"Sharp" if np.std(y) > y.mean() else "Broad"}
    """
    
    model_prompt = f"""
    Based on this spectrum data, select the most appropriate lmfit model type:
    {spectrum_summary}
    
    Available models:
    - gaussian: For symmetric, bell-shaped peaks
    - lorentzian: For broader, more rounded peaks  
    - voigt: For peaks with both Gaussian and Lorentzian character
    - pseudovoigt: Similar to Voigt but computationally faster
    - skewed_gaussian: For asymmetric peaks
    - skewed_voigt: For asymmetric peaks with mixed character
    
    Return ONLY the model name (e.g., "gaussian" or "voigt").
    """
    
    try:
        response = llm.generate(model_prompt, max_tokens=50)
        # Extract model name from response
        model_name = response.strip().lower()
        
        # Validate model name
        valid_models = ['gaussian', 'lorentzian', 'voigt', 'pseudovoigt', 'skewed_gaussian', 'skewed_voigt']
        if model_name in valid_models:
            return model_name
        else:
            print(f"LLM returned invalid model '{model_name}', defaulting to gaussian")
            return "gaussian"
    except Exception as e:
        print(f"Error in model selection: {e}, defaulting to gaussian")
        return "gaussian"


def run_complete_analysis(
    config: CurveFittingConfig,
    well_name: str,
    llm: LLMClient,
    read: int = 1,
    max_peaks: int = 3,
    model_kind: Optional[str] = None,
    r2_target: float = 0.90,
    max_attempts: int = 3
) -> Dict[str, object]:
    """Run complete analysis workflow for a single well."""
    # Get data
    x, y = get_xy_for_well(config, well_name, read=read)
    
    # Let LLM select model type if not specified
    if model_kind is None:
        print(f"  Selecting model type for {well_name}...")
        model_kind = select_model_for_spectrum(llm, x, y, well_name)
        print(f"  LLM selected model: {model_kind}")
    
    # LLM numeric analysis
    sys_prompt_numeric = get_prompt("numeric")
    llm_result = llm_guess_peaks(llm, x, y, use_image=False, system_prompt=sys_prompt_numeric, max_peaks=max_peaks)
    
    # Create output folder for images
    os.makedirs("analysis_output", exist_ok=True)
    
    # Save spectrum image and analyze with LLM
    spectrum_image_path = f'analysis_output/spectrum_{well_name}.png'
    save_plot_png(x, y, spectrum_image_path, title=f'PL Spectrum - Well {well_name}')
    
    sys_prompt_image = get_prompt("image")
    llm_image_result = llm_guess_peaks_from_image(llm, spectrum_image_path, system_prompt=sys_prompt_image, max_peaks=max_peaks)
    
    # lmfit fitting with automatic model retry for poor fits
    fit_result = fit_peaks_lmfit_with_retry(
        x, y, llm_result, 
        model_kind=model_kind,
        r2_target=0.92,  # Higher target
        max_attempts=max_attempts
    )
    
    # Save initial fitting plot for LLM assessment
    temp_plot_path = f'analysis_output/temp_fit_{well_name}.png'
    save_fitting_plot_png(x, y, fit_result, temp_plot_path, 
                          title=f'Initial Fit - Well {well_name}')
    
    # LLM visual assessment of fit quality
    llm_assessment = assess_fit_quality_with_llm(llm, temp_plot_path, fit_result.stats.r2, well_name)
    
    # Retry logic: continue until R² > 0.9 AND LLM says it's good
    retry_count = 0
    max_retries = 8  # Increased retry attempts
    alternative_models = ['gaussian', 'voigt', 'lorentzian', 'pseudovoigt', 'skewed_gaussian']
    if model_kind in alternative_models:
        alternative_models.remove(model_kind)  # Don't retry the same model first
    
    best_fit = fit_result
    
    while (fit_result.stats.r2 < 0.90 or not llm_assessment) and retry_count < max_retries:
        retry_count += 1
        print(f"  Attempt {retry_count}: R²={fit_result.stats.r2:.3f}, LLM assessment={'good' if llm_assessment else 'poor'}")
        
        # Try alternative models
        alt_model = alternative_models[retry_count % len(alternative_models)]
        try:
            alt_result = fit_peaks_lmfit_with_retry(
                x, y, llm_result, 
                model_kind=alt_model,
                r2_target=0.92,
                max_attempts=3
            )
            
            if alt_result and alt_result.stats.r2 > best_fit.stats.r2:
                # Test this alternative with LLM
                save_fitting_plot_png(x, y, alt_result, temp_plot_path, 
                                     title=f'Alternative Fit ({alt_model}) - Well {well_name}')
                alt_assessment = assess_fit_quality_with_llm(llm, temp_plot_path, alt_result.stats.r2, well_name)
                
                if alt_result.stats.r2 > 0.90 and alt_assessment:
                    print(f"  Success with {alt_model}: R²={alt_result.stats.r2:.3f}, LLM=good")
                    best_fit = alt_result
                    break
                elif alt_result.stats.r2 > best_fit.stats.r2:
                    print(f"  Improved with {alt_model}: R²={alt_result.stats.r2:.3f}")
                    best_fit = alt_result
                    
        except Exception as e:
            print(f"  Alternative model {alt_model} failed: {e}")
            continue
        
        fit_result = best_fit
        llm_assessment = assess_fit_quality_with_llm(llm, temp_plot_path, fit_result.stats.r2, well_name)
    
    # Clean up temp file
    try:
        os.remove(temp_plot_path)
    except:
        pass
    
    fit_result = best_fit
    
    # Save fitting plot in output folder
    fitting_plot_file = save_fitting_plot_png(x, y, fit_result, f'analysis_output/fit_results_{well_name}.png', 
                                             title=f'Peak Fitting Results - Well {well_name}')
    
    # Quality assessment
    quality_assessment = assess_fitting_quality(fit_result)
    
    return {
        'well_name': well_name,
        'read': read,
        'data': {'x': x, 'y': y},
        'llm_numeric_result': llm_result,
        'llm_image_result': llm_image_result,
        'fit_result': fit_result,
        'files': {
            'spectrum_image': spectrum_image_path,
            'fitting_plot': fitting_plot_file
        },
        'quality_assessment': quality_assessment
    }


# ---------- LLM peak guessing helpers ----------

def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return x[idx], y[idx]


def _extract_json(text: str) -> Dict[str, Any]:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)
    a = s.find("{")
    b = s.rfind("}")
    if a != -1 and b != -1 and b > a:
        return json.loads(s[a : b + 1])
    raise ValueError("No JSON object found in LLM response.")


def build_peak_prompt_from_series(
    x: Iterable[float],
    y: Iterable[float],
    *,
    max_peaks: int = 5,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Model must return ONLY JSON:
    {
      "peaks": [{"center": float, "height": float, "fwhm": float|null, "prominence": float|null}, ...],
      "baseline": float|null
    }
    """
    header = (system_prompt.strip() + "\n\n") if system_prompt else \
        "Return ONLY JSON with keys 'peaks' (list) and 'baseline'. No prose.\n"
    series = {
        "x": list(map(float, x)),
        "y": list(map(float, y)),
        "max_peaks": int(max_peaks),
        "fields": ["center", "height", "fwhm", "prominence"],
    }
    return header + "Series:\n" + json.dumps(series)


def llm_guess_peaks_from_data(
    llm: LLMClient,
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_peaks: int = 5,
    system_prompt: Optional[str] = None,
    max_tokens: int = 500,
) -> PeakResult:
    xs, ys = _downsample_xy(np.asarray(x), np.asarray(y), max_points=512)
    prompt = build_peak_prompt_from_series(xs, ys, max_peaks=max_peaks, system_prompt=system_prompt)
    text = llm.generate(prompt, max_tokens=max_tokens)
    obj = _extract_json(text)
    peaks = [
        PeakGuess(
            center=float(p.get("center")),
            height=float(p.get("height")),
            fwhm=(float(p["fwhm"]) if p.get("fwhm") is not None else None),
            prominence=(float(p["prominence"]) if p.get("prominence") is not None else None),
        )
        for p in obj.get("peaks", [])
    ]
    base = obj.get("baseline")
    return PeakResult(peaks=peaks, baseline=(float(base) if base is not None else None))


def llm_guess_peaks_from_image(
    llm: LLMClient,
    image_path: str,
    *,
    max_peaks: int = 5,
    system_prompt: Optional[str] = None,
    max_tokens: int = 500,
) -> PeakResult:
    if genai is None:
        raise RuntimeError("google-generativeai is required for image analysis.")
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required to load images.")

    instr = ((system_prompt.strip() + "\n\n") if system_prompt else "") + \
        ("Return ONLY JSON with keys 'peaks' (list) and 'baseline'. No prose.\n"
         f"Max peaks: {int(max_peaks)}\n")

    img = Image.open(image_path)
    try:
        text = llm.generate_multimodal([instr, img], max_tokens=max_tokens)
    finally:
        try:
            img.close()
        except Exception:
            pass

    obj = _extract_json(text)
    peaks = [
        PeakGuess(
            center=float(p.get("center")),
            height=float(p.get("height")),
            fwhm=(float(p["fwhm"]) if p.get("fwhm") is not None else None),
            prominence=(float(p["prominence"]) if p.get("prominence") is not None else None),
        )
        for p in obj.get("peaks", [])
    ]
    base = obj.get("baseline")
    return PeakResult(peaks=peaks, baseline=(float(base) if base is not None else None))


def llm_guess_peaks(
    llm: LLMClient,
    x: np.ndarray,
    y: np.ndarray,
    *,
    use_image: bool = False,
    image_path: Optional[str] = None,
    max_peaks: int = 5,
    system_prompt: Optional[str] = None,
    max_tokens: int = 500,
    temp_plot_path: str = "_tmp_curve.png",
) -> PeakResult:
    if use_image:
        if image_path is None:
            save_plot_png(x, y, temp_plot_path)
            image_path = temp_plot_path
        return llm_guess_peaks_from_image(
            llm, image_path, max_peaks=max_peaks, system_prompt=system_prompt, max_tokens=max_tokens
        )
    return llm_guess_peaks_from_data(
        llm, x, y, max_peaks=max_peaks, system_prompt=system_prompt, max_tokens=max_tokens
    )


# ---------- lmfit multi-peak fitting with retry ----------

try:
    import lmfit
    from lmfit.models import (
        ConstantModel, GaussianModel, VoigtModel, LorentzianModel, 
        PseudoVoigtModel, SkewedGaussianModel, SkewedVoigtModel,
        ExponentialGaussianModel, SplitLorentzianModel
    )  # type: ignore
    _HAS_LMFIT = True
except Exception:
    _HAS_LMFIT = False


@dataclass
class PeakFitStats:
    r2: float
    rmse: float
    aic: float
    bic: float
    redchi: float
    nfev: int


@dataclass
class PeakFitResult:
    success: bool
    stats: PeakFitStats
    best_params: Dict[str, float]
    peaks: List[PeakGuess]          # updated centers/heights/FWHM after fit
    baseline: Optional[float]
    report: str
    model_kind: str                 # 'gaussian' or 'voigt'


def _estimate_sigma_from_fwhm(fwhm: float) -> float:
    return float(fwhm) / 2.354820045  # FWHM = 2*sqrt(2*ln2)*sigma


def _guess_sigma_from_span(x: np.ndarray, frac: float = 0.02) -> float:
    span = float(np.max(x) - np.min(x))
    return max(1e-6, frac * span)


def _height_to_gaussian_amplitude(height: float, sigma: float) -> float:
    # For GaussianModel, amplitude is area = height * sigma * sqrt(2*pi)
    return float(height) * float(sigma) * np.sqrt(2.0 * np.pi)


def _gaussian_height_from_amp(amplitude: float, sigma: float) -> float:
    return float(amplitude) / (float(sigma) * np.sqrt(2.0 * np.pi))


def _build_composite_model(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[PeakGuess],
    baseline: Optional[float],
    *,
    model_kind: str = "gaussian",       # 'gaussian' or 'voigt'
    center_window: Optional[float] = None,   # absolute window around initial center
    sigma_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[lmfit.Model, lmfit.Parameters]:
    if not _HAS_LMFIT:
        raise RuntimeError("lmfit is required. pip install lmfit")

    xmin, xmax = float(np.min(x)), float(np.max(x))
    span = xmax - xmin
    if center_window is None:
        center_window = 0.10 * span
    if sigma_bounds is None:
        sigma_bounds = (max(1e-6, 0.002 * span), max(1e-6, 0.20 * span))

    model = ConstantModel(prefix="c_")
    params = model.make_params()
    base_val = float(baseline) if baseline is not None else float(np.nanmin(y))
    params["c_c"].set(value=base_val, min=-np.inf, max=np.inf)

    for i, pk in enumerate(peaks):
        prefix = f"p{i}_"
        comp = select_peak_model(model_kind)(prefix=prefix)
        model = model + comp

        if pk.fwhm and pk.fwhm > 0:
            sigma0 = _estimate_sigma_from_fwhm(pk.fwhm)
        else:
            sigma0 = _guess_sigma_from_span(x)

        amp0 = _height_to_gaussian_amplitude(pk.height, sigma0) if pk.height is not None else \
            float(np.trapz(y, x) / max(1, len(peaks)))

        p = comp.make_params()
        p[f"{prefix}amplitude"].set(value=float(amp0), min=0.0, max=np.inf)
        c0 = float(pk.center)
        p[f"{prefix}center"].set(value=c0, min=c0 - center_window, max=c0 + center_window)
        p[f"{prefix}sigma"].set(value=float(sigma0), min=sigma_bounds[0], max=sigma_bounds[1])

        if model_kind == "voigt" and f"{prefix}gamma" in p:
            p[f"{prefix}gamma"].set(value=float(sigma0), min=sigma_bounds[0], max=sigma_bounds[1])

        params.update(p)

    return model, params


def _fit_once(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[PeakGuess],
    baseline: Optional[float],
    *,
    model_kind: str = "gaussian",
) -> lmfit.model.ModelResult:
    model, params = _build_composite_model(x, y, peaks, baseline, model_kind=model_kind)
    return model.fit(y, params, x=x, nan_policy="omit", max_nfev=10000)


def _score_fit(y: np.ndarray, yhat: np.ndarray) -> Tuple[float, float]:
    resid = y - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) + 1e-16
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(ss_res / max(1, len(y))))
    return r2, rmse


def _result_to_peaks(result: lmfit.model.ModelResult, model_kind: str) -> Tuple[List[PeakGuess], Optional[float], Dict[str, float]]:
    params = result.best_values
    baseline = params.get("c_c", None)

    out: List[PeakGuess] = []
    i = 0
    while True:
        # Try different prefix patterns used by lmfit
        prefixes_to_try = [f"p{i}_", f"g{i}_", f"v{i}_", f"l{i}_"]
        found_peak = False
        
        for prefix in prefixes_to_try:
            amp_key = f"{prefix}amplitude"
            center_key = f"{prefix}center"
            sigma_key = f"{prefix}sigma"
            
            if amp_key in params and center_key in params and sigma_key in params:
                amp = float(params[amp_key])
                cen = float(params[center_key])
                sig = float(params[sigma_key])
                
                # Calculate height and FWHM based on model type
                if model_kind in ['gaussian', 'skewed_gaussian']:
                    height = _gaussian_height_from_amp(amp, sig)
                    fwhm = 2.354820045 * sig
                elif model_kind in ['lorentzian', 'split_lorentzian']:
                    height = amp / (sig * np.pi)  # Lorentzian height from amplitude
                    fwhm = 2.0 * sig
                elif model_kind in ['voigt', 'pseudovoigt', 'skewed_voigt']:
                    # For Voigt, use gamma parameter if available
                    gamma_key = f"{prefix}gamma"
                    if gamma_key in params:
                        gamma = float(params[gamma_key])
                        fwhm = 3.6013 * gamma  # Approximation for Voigt FWHM
                    else:
                        fwhm = 2.354820045 * sig  # Fallback to Gaussian
                    height = _gaussian_height_from_amp(amp, sig)  # Approximation
                else:
                    height = _gaussian_height_from_amp(amp, sig)
                    fwhm = 2.354820045 * sig
                
                out.append(PeakGuess(center=cen, height=height, fwhm=fwhm, prominence=None))
                found_peak = True
                break
        
        if not found_peak:
            break
        i += 1

    return out, (float(baseline) if baseline is not None else None), {k: float(v) for k, v in params.items()}


def fit_peaks_lmfit_with_retry(
    x: np.ndarray,
    y: np.ndarray,
    seed: PeakResult,
    *,
    model_kind: str = "gaussian",    # 'gaussian' or 'voigt'
    r2_target: float = 0.90,
    max_attempts: int = 5,
    jitter_frac_center: float = 0.01,
    jitter_frac_sigma: float = 0.25,
    allow_baseline_refit: bool = True,
) -> PeakFitResult:
    """
    Fit multi-peak PL curves with lmfit. Retries with randomized restarts
    until R² >= r2_target or attempts are exhausted.
    """
    if not _HAS_LMFIT:
        raise RuntimeError("lmfit is required. pip install lmfit")

    rng = np.random.default_rng()
    xmin, xmax = float(np.min(x)), float(np.max(x))
    span = xmax - xmin

    def _jitter(pr: PeakResult) -> PeakResult:
        new_peaks: List[PeakGuess] = []
        for pk in pr.peaks:
            c = pk.center + rng.uniform(-jitter_frac_center, jitter_frac_center) * span
            if pk.fwhm and pk.fwhm > 0:
                sig = _estimate_sigma_from_fwhm(pk.fwhm)
                sig *= (1.0 + rng.uniform(-jitter_frac_sigma, jitter_frac_sigma))
                fwhm = 2.354820045 * max(sig, 1e-6)
            else:
                fwhm = None
            h = max(1e-12, pk.height * (1.0 + rng.uniform(-0.15, 0.15)))
            new_peaks.append(PeakGuess(center=float(c), height=float(h), fwhm=(float(fwhm) if fwhm else None)))
        base = pr.baseline
        if allow_baseline_refit and base is not None:
            base = base * (1.0 + rng.uniform(-0.10, 0.10))
        return PeakResult(peaks=new_peaks, baseline=(float(base) if base is not None else None))

    best_result = None
    best_stats = None
    best_report = ""
    best_params: Dict[str, float] = {}
    best_peaks: List[PeakGuess] = []
    success = False

    attempt = 0
    current_seed = seed

    while attempt < max_attempts:
        attempt += 1
        try:
            result = _fit_once(x, y, current_seed.peaks, current_seed.baseline, model_kind=model_kind)
        except Exception:
            current_seed = _jitter(seed)
            continue

        yhat = result.best_fit
        r2, rmse = _score_fit(y, yhat)
        stats = PeakFitStats(
            r2=r2,
            rmse=rmse,
            aic=float(result.aic),
            bic=float(result.bic),
            redchi=float(result.redchi),
            nfev=int(result.nfev),
        )
        peaks_out, base_out, params_out = _result_to_peaks(result, model_kind)

        if (best_result is None) or (stats.r2 > best_stats.r2):  # type: ignore
            best_result = result
            best_stats = stats
            best_report = result.fit_report(min_correl=0.3)
            best_params = params_out
            best_peaks = peaks_out

        if stats.r2 >= r2_target:
            success = True
            break

        current_seed = _jitter(seed)

    if best_result is None or best_stats is None:
        raise RuntimeError("lmfit did not produce a valid result across attempts")

    return PeakFitResult(
        success=success,
        stats=best_stats,
        best_params=best_params,
        peaks=best_peaks,
        baseline=best_params.get("c_c", None),
        report=best_report,
        model_kind=model_kind,
    )


# ---------- Dataset curation (reads/wells/wavelengths) ----------

@dataclass
class CurveFittingConfig:
    data_csv: Optional[str] = None
    composition_csv: Optional[str] = None

    start_wavelength: Optional[int] = None
    end_wavelength: Optional[int] = None
    wavelength_step_size: Optional[int] = None

    read_selection: Union[str, Iterable[int], None] = "all"
    wells_to_ignore: Union[str, Iterable[str], None] = None

    fill_na_value: float = 0.0

    @staticmethod
    def _parse_int_list(text: str) -> List[int]:
        items: List[int] = []
        for chunk in re.split(r"\s*,\s*", text.strip()):
            if not chunk:
                continue
            if re.match(r"^\d+-\d+$", chunk):
                a, b = map(int, chunk.split("-"))
                if a > b:
                    a, b = b, a
                items.extend(range(a, b + 1))
            elif chunk.isdigit():
                items.append(int(chunk))
            else:
                raise ValueError(f"Invalid integer/range token: '{chunk}'")
        return sorted(set(items))

    @staticmethod
    def _parse_str_list(text: str) -> List[str]:
        parts = [p.strip().upper() for p in re.split(r"\s*,\s*", text.strip()) if p.strip()]
        for p in parts:
            if not re.match(r"^[A-H](?:[1-9]|1[0-2])$", p):
                raise ValueError(f"Invalid well id: '{p}' (expected A1..H12)")
        return parts

    @classmethod
    def from_user_inputs(
        cls,
        data_csv: str,
        composition_csv: str,
        read_selection: Union[str, Iterable[int], None] = "all",
        wells_to_ignore: Union[str, Iterable[str], None] = None,
        start_wavelength: Optional[int] = None,
        end_wavelength: Optional[int] = None,
        wavelength_step_size: Optional[int] = None,
        fill_na_value: float = 0.0,
    ) -> CurveFittingConfig:
        if isinstance(read_selection, str) and read_selection.lower() != "all":
            read_selection = cls._parse_int_list(read_selection)
        elif isinstance(read_selection, Iterable) and not isinstance(read_selection, (str, bytes)):
            read_selection = list(map(int, read_selection))

        if isinstance(wells_to_ignore, str):
            wells_to_ignore = cls._parse_str_list(wells_to_ignore)
        elif wells_to_ignore is None:
            wells_to_ignore = []
        else:
            wells_to_ignore = [str(w).strip().upper() for w in wells_to_ignore]

        return cls(
            data_csv=data_csv,
            composition_csv=composition_csv,
            start_wavelength=start_wavelength,
            end_wavelength=end_wavelength,
            wavelength_step_size=wavelength_step_size,
            read_selection=read_selection,
            wells_to_ignore=wells_to_ignore,
            fill_na_value=fill_na_value,
        )


class CurveFitting:
    READ_HEADER_PATTERN = re.compile(r"^Read\s+(\d+):EM\s+Spectrum\s*$", re.I)

    def __init__(self, config: CurveFittingConfig):
        self.cfg = config

    @staticmethod
    def load_csvs(data_path: str, comp_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not (data_path.lower().endswith(".csv") and comp_path.lower().endswith(".csv")):
            raise ValueError("Both data_path and comp_path must be .csv files")
        data = pd.read_csv(data_path, header=None)
        data = data.replace("OVRFLW", np.nan)
        composition = pd.read_csv(comp_path, index_col=0)
        return data, composition

    @classmethod
    def _find_read_block_starts(cls, data: pd.DataFrame) -> Dict[int, int]:
        starts: Dict[int, int] = {}
        first_col = data.iloc[:, 0].astype(str)
        for idx, val in first_col.items():
            m = cls.READ_HEADER_PATTERN.match(val)
            if m:
                starts[int(m.group(1))] = idx
        if not starts:
            raise ValueError("No 'Read N:EM Spectrum' blocks found in data CSV")
        return dict(sorted(starts.items()))

    @staticmethod
    def _slice_block(data: pd.DataFrame, start_row: int, end_row: Optional[int]) -> pd.DataFrame:
        end = end_row if end_row is not None else len(data)
        block = data.iloc[start_row + 2 : end - 1].copy()
        if 0 in block.columns:
            block = block.drop(columns=[0])
        new_header = block.iloc[0]
        block = block.iloc[1:]
        block.columns = new_header
        block = block.apply(pd.to_numeric, errors="coerce")
        return block

    @classmethod
    def parse_all_reads(cls, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        starts = cls._find_read_block_starts(data)
        read_indices = list(starts.keys())
        blocks: Dict[int, pd.DataFrame] = {}
        for i, r in enumerate(read_indices):
            start_row = starts[r]
            end_row = starts[read_indices[i + 1]] if i + 1 < len(read_indices) else None
            blocks[r] = cls._slice_block(data, start_row, end_row)
        return blocks

    @staticmethod
    def select_reads(blocks: Dict[int, pd.DataFrame], selection: Union[str, Iterable[int], None]) -> Dict[int, pd.DataFrame]:
        if selection is None or (isinstance(selection, str) and selection.lower() == "all"):
            return dict(sorted(blocks.items()))
        desired = sorted(set(map(int, selection)))
        return {k: v for k, v in blocks.items() if k in desired}

    @staticmethod
    def drop_wells(blocks: Dict[int, pd.DataFrame], wells_to_ignore: Iterable[str]) -> Dict[int, pd.DataFrame]:
        wells_to_ignore = [w.strip().upper() for w in wells_to_ignore or []]
        if not wells_to_ignore:
            return blocks
        cleaned: Dict[int, pd.DataFrame] = {}
        for k, df in blocks.items():
            keep_cols = [c for c in df.columns if str(c).strip().upper() not in wells_to_ignore]
            cleaned[k] = df[keep_cols]
        return cleaned

    @staticmethod
    def infer_wavelength_vector(df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        return np.arange(n)

    def build_wavelengths(self, exemplar_df: pd.DataFrame) -> np.ndarray:
        cfg = self.cfg
        if cfg.start_wavelength is not None and cfg.end_wavelength is not None and cfg.wavelength_step_size is not None:
            return np.arange(cfg.start_wavelength, cfg.end_wavelength + cfg.wavelength_step_size, cfg.wavelength_step_size)
        return self.infer_wavelength_vector(exemplar_df)

    def stack_blocks(self, blocks: Dict[int, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        well_sets = [set(map(str, df.columns)) for df in blocks.values()]
        common_wells = sorted(set.intersection(*well_sets)) if well_sets else []
        if not common_wells:
            common_wells = sorted(set.union(*well_sets)) if well_sets else []

        read_indices = sorted(blocks.keys())
        exemplar = blocks[read_indices[0]]
        x = self.build_wavelengths(exemplar)
        num_reads = len(read_indices)
        num_wavelengths = len(exemplar)
        num_wells = len(common_wells)

        tensor = np.full((num_reads, num_wavelengths, num_wells), fill_value=np.nan, dtype=float)
        for i, r in enumerate(read_indices):
            df = blocks[r].reindex(columns=common_wells)
            arr = df.to_numpy(dtype=float)
            tensor[i, : arr.shape[0], : arr.shape[1]] = arr

        tensor = np.nan_to_num(tensor, nan=self.cfg.fill_na_value)
        return tensor, x, common_wells, read_indices

    def curate_dataset(self) -> Dict[str, object]:
        if not self.cfg.data_csv or not self.cfg.composition_csv:
            raise ValueError("Both data_csv and composition_csv must be provided.")

        raw_data, composition = self.load_csvs(self.cfg.data_csv, self.cfg.composition_csv)
        all_blocks = self.parse_all_reads(raw_data)
        sel_blocks = self.select_reads(all_blocks, self.cfg.read_selection)
        if not sel_blocks:
            raise ValueError("No reads selected; check read_selection.")
        sel_blocks = self.drop_wells(sel_blocks, self.cfg.wells_to_ignore)
        tensor, wavelengths, wells, reads = self.stack_blocks(sel_blocks)

        comp_aligned = composition.copy()
        comp_aligned = comp_aligned.loc[:, [w for w in wells if w in comp_aligned.columns]]

        return {
            "tensor": tensor,
            "wavelengths": wavelengths,
            "wells": wells,
            "reads": reads,
            "composition": comp_aligned,
            "blocks": sel_blocks,
        }

    def get_xy(self, curated: Dict[str, object], well: str, read: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        well = str(well).strip().upper()
        wells: List[str] = curated["wells"]  # type: ignore
        if well not in wells:
            raise KeyError(f"Well '{well}' not in curated set: {wells}")
        reads: List[int] = curated["reads"]  # type: ignore
        if read is None:
            read = reads[0]
        if read not in reads:
            raise KeyError(f"Read {read} not in curated reads: {reads}")
        w_idx = wells.index(well)
        r_idx = reads.index(read)
        x = curated["wavelengths"]  # type: ignore
        y = curated["tensor"][r_idx, :, w_idx]  # type: ignore
        return x, y


# ---------- Agent-callable helpers ----------

def build_agent_config(
    data_csv: str,
    composition_csv: str,
    read_selection: Union[str, Iterable[int], None] = "all",
    wells_to_ignore: Union[str, Iterable[str], None] = None,
    start_wavelength: Optional[int] = None,
    end_wavelength: Optional[int] = None,
    wavelength_step_size: Optional[int] = None,
    fill_na_value: float = 0.0,
) -> CurveFittingConfig:
    return CurveFittingConfig.from_user_inputs(
        data_csv=data_csv,
        composition_csv=composition_csv,
        read_selection=read_selection,
        wells_to_ignore=wells_to_ignore,
        start_wavelength=start_wavelength,
        end_wavelength=end_wavelength,
        wavelength_step_size=wavelength_step_size,
        fill_na_value=fill_na_value,
    )


def curate_dataset(config: CurveFittingConfig) -> Dict[str, object]:
    agent = CurveFitting(config)
    return agent.curate_dataset()


def get_xy_for_well(config: CurveFittingConfig, well: str, read: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    agent = CurveFitting(config)
    curated = agent.curate_dataset()
    return agent.get_xy(curated, well=well, read=read)


# ---------- lmfit multi-peak fitting with retry ----------

try:
    import lmfit
    from lmfit.models import (
        ConstantModel, GaussianModel, VoigtModel, LorentzianModel, 
        PseudoVoigtModel, SkewedGaussianModel, SkewedVoigtModel,
        ExponentialGaussianModel, SplitLorentzianModel
    )  # type: ignore
    _HAS_LMFIT = True
except Exception:
    _HAS_LMFIT = False


@dataclass
class PeakFitStats:
    r2: float
    rmse: float
    aic: float
    bic: float
    redchi: float
    nfev: int


@dataclass
class PeakFitResult:
    success: bool
    stats: PeakFitStats
    best_params: Dict[str, float]
    peaks: List[PeakGuess]          # updated centers/heights/FWHM after fit
    baseline: Optional[float]
    report: str
    model_kind: str                 # 'gaussian' or 'voigt'


def _estimate_sigma_from_fwhm(fwhm: float) -> float:
    return float(fwhm) / 2.354820045  # FWHM = 2*sqrt(2*ln2)*sigma


def _guess_sigma_from_span(x: np.ndarray, frac: float = 0.02) -> float:
    span = float(np.max(x) - np.min(x))
    return max(1e-6, frac * span)


def _height_to_gaussian_amplitude(height: float, sigma: float) -> float:
    # For GaussianModel, amplitude is area = height * sigma * sqrt(2*pi)
    return float(height) * float(sigma) * np.sqrt(2.0 * np.pi)


def _gaussian_height_from_amp(amplitude: float, sigma: float) -> float:
    return float(amplitude) / (float(sigma) * np.sqrt(2.0 * np.pi))


def _build_composite_model(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[PeakGuess],
    baseline: Optional[float],
    *,
    model_kind: str = "gaussian",       # 'gaussian' or 'voigt'
    center_window: Optional[float] = None,   # absolute window around initial center
    sigma_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[lmfit.Model, lmfit.Parameters]:
    if not _HAS_LMFIT:
        raise RuntimeError("lmfit is required. pip install lmfit")

    xmin, xmax = float(np.min(x)), float(np.max(x))
    span = xmax - xmin
    if center_window is None:
        center_window = 0.10 * span
    if sigma_bounds is None:
        sigma_bounds = (max(1e-6, 0.002 * span), max(1e-6, 0.20 * span))

    model = ConstantModel(prefix="c_")
    params = model.make_params()
    base_val = float(baseline) if baseline is not None else float(np.nanmin(y))
    params["c_c"].set(value=base_val, min=-np.inf, max=np.inf)

    for i, pk in enumerate(peaks):
        prefix = f"p{i}_"
        comp = select_peak_model(model_kind)(prefix=prefix)
        model = model + comp

        if pk.fwhm and pk.fwhm > 0:
            sigma0 = _estimate_sigma_from_fwhm(pk.fwhm)
        else:
            sigma0 = _guess_sigma_from_span(x)

        amp0 = _height_to_gaussian_amplitude(pk.height, sigma0) if pk.height is not None else \
            float(np.trapz(y, x) / max(1, len(peaks)))

        p = comp.make_params()
        p[f"{prefix}amplitude"].set(value=float(amp0), min=0.0, max=np.inf)
        c0 = float(pk.center)
        p[f"{prefix}center"].set(value=c0, min=c0 - center_window, max=c0 + center_window)
        p[f"{prefix}sigma"].set(value=float(sigma0), min=sigma_bounds[0], max=sigma_bounds[1])

        if model_kind == "voigt" and f"{prefix}gamma" in p:
            p[f"{prefix}gamma"].set(value=float(sigma0), min=sigma_bounds[0], max=sigma_bounds[1])

        params.update(p)

    return model, params


def _fit_once(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[PeakGuess],
    baseline: Optional[float],
    *,
    model_kind: str = "gaussian",
) -> lmfit.model.ModelResult:
    model, params = _build_composite_model(x, y, peaks, baseline, model_kind=model_kind)
    return model.fit(y, params, x=x, nan_policy="omit", max_nfev=10000)


def _score_fit(y: np.ndarray, yhat: np.ndarray) -> Tuple[float, float]:
    resid = y - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) + 1e-16
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(ss_res / max(1, len(y))))
    return r2, rmse


def _result_to_peaks(result: lmfit.model.ModelResult, model_kind: str) -> Tuple[List[PeakGuess], Optional[float], Dict[str, float]]:
    params = result.best_values
    baseline = params.get("c_c", None)

    out: List[PeakGuess] = []
    i = 0
    while True:
        # Try different prefix patterns used by lmfit
        prefixes_to_try = [f"p{i}_", f"g{i}_", f"v{i}_", f"l{i}_"]
        found_peak = False
        
        for prefix in prefixes_to_try:
            amp_key = f"{prefix}amplitude"
            center_key = f"{prefix}center"
            sigma_key = f"{prefix}sigma"
            
            if amp_key in params and center_key in params and sigma_key in params:
                amp = float(params[amp_key])
                cen = float(params[center_key])
                sig = float(params[sigma_key])
                
                # Calculate height and FWHM based on model type
                if model_kind in ['gaussian', 'skewed_gaussian']:
                    height = _gaussian_height_from_amp(amp, sig)
                    fwhm = 2.354820045 * sig
                elif model_kind in ['lorentzian', 'split_lorentzian']:
                    height = amp / (sig * np.pi)  # Lorentzian height from amplitude
                    fwhm = 2.0 * sig
                elif model_kind in ['voigt', 'pseudovoigt', 'skewed_voigt']:
                    # For Voigt, use gamma parameter if available
                    gamma_key = f"{prefix}gamma"
                    if gamma_key in params:
                        gamma = float(params[gamma_key])
                        fwhm = 3.6013 * gamma  # Approximation for Voigt FWHM
                    else:
                        fwhm = 2.354820045 * sig  # Fallback to Gaussian
                    height = _gaussian_height_from_amp(amp, sig)  # Approximation
                else:
                    height = _gaussian_height_from_amp(amp, sig)
                    fwhm = 2.354820045 * sig
                
                out.append(PeakGuess(center=cen, height=height, fwhm=fwhm, prominence=None))
                found_peak = True
                break
        
        if not found_peak:
            break
        i += 1

    return out, (float(baseline) if baseline is not None else None), {k: float(v) for k, v in params.items()}


def fit_peaks_lmfit_with_retry(
    x: np.ndarray,
    y: np.ndarray,
    seed: PeakResult,
    *,
    model_kind: str = "gaussian",    # 'gaussian' or 'voigt'
    r2_target: float = 0.90,
    max_attempts: int = 5,
    jitter_frac_center: float = 0.01,
    jitter_frac_sigma: float = 0.25,
    allow_baseline_refit: bool = True,
) -> PeakFitResult:
    """
    Fit multi-peak PL curves with lmfit. Retries with randomized restarts
    until R² >= r2_target or attempts are exhausted.
    """
    if not _HAS_LMFIT:
        raise RuntimeError("lmfit is required. pip install lmfit")

    rng = np.random.default_rng()
    xmin, xmax = float(np.min(x)), float(np.max(x))
    span = xmax - xmin

    def _jitter(pr: PeakResult) -> PeakResult:
        new_peaks: List[PeakGuess] = []
        for pk in pr.peaks:
            c = pk.center + rng.uniform(-jitter_frac_center, jitter_frac_center) * span
            if pk.fwhm and pk.fwhm > 0:
                sig = _estimate_sigma_from_fwhm(pk.fwhm)
                sig *= (1.0 + rng.uniform(-jitter_frac_sigma, jitter_frac_sigma))
                fwhm = 2.354820045 * max(sig, 1e-6)
            else:
                fwhm = None
            h = max(1e-12, pk.height * (1.0 + rng.uniform(-0.15, 0.15)))
            new_peaks.append(PeakGuess(center=float(c), height=float(h), fwhm=(float(fwhm) if fwhm else None)))
        base = pr.baseline
        if allow_baseline_refit and base is not None:
            base = base * (1.0 + rng.uniform(-0.10, 0.10))
        return PeakResult(peaks=new_peaks, baseline=(float(base) if base is not None else None))

    best_result = None
    best_stats = None
    best_report = ""
    best_params: Dict[str, float] = {}
    best_peaks: List[PeakGuess] = []
    success = False

    attempt = 0
    current_seed = seed

    while attempt < max_attempts:
        attempt += 1
        try:
            result = _fit_once(x, y, current_seed.peaks, current_seed.baseline, model_kind=model_kind)
        except Exception:
            current_seed = _jitter(seed)
            continue

        yhat = result.best_fit
        r2, rmse = _score_fit(y, yhat)
        stats = PeakFitStats(
            r2=r2,
            rmse=rmse,
            aic=float(result.aic),
            bic=float(result.bic),
            redchi=float(result.redchi),
            nfev=int(result.nfev),
        )
        peaks_out, base_out, params_out = _result_to_peaks(result, model_kind)

        if (best_result is None) or (stats.r2 > best_stats.r2):  # type: ignore
            best_result = result
            best_stats = stats
            best_report = result.fit_report(min_correl=0.3)
            best_params = params_out
            best_peaks = peaks_out

        if stats.r2 >= r2_target:
            success = True
            break

        current_seed = _jitter(seed)

    if best_result is None or best_stats is None:
        raise RuntimeError("lmfit did not produce a valid result across attempts")

    return PeakFitResult(
        success=success,
        stats=best_stats,
        best_params=best_params,
        peaks=best_peaks,
        baseline=best_params.get("c_c", None),
        report=best_report,
        model_kind=model_kind,
    )
