# prompts.py (or paste at top of fitting_agent.py)

from __future__ import annotations

from typing import Dict

PROMPTS: Dict[str, str] = {
    # Use with llm_guess_peaks(..., use_image=False)
    "numeric": """You receive a photoluminescence spectrum as numeric arrays x (wavelength) and y (intensity).
Return ONLY one JSON object. No explanations, no markdown.

Task:
- Identify up to max_peaks emission peaks.
- Produce initial guesses suitable for non-linear fitting.

Output JSON schema:
{
  "peaks": [
    { "center": number, "height": number, "fwhm": number|null, "prominence": number|null }
  ],
  "baseline": number|null
}

Rules:
- Centers within the x range.
- Heights ≥ 0.
- If baseline or FWHM are uncertain, set null.
- Do not exceed max_peaks.
- Output must be valid JSON.

Input follows under "Series:" as JSON with keys: x, y, max_peaks, fields.
""",

    # Use with llm_guess_peaks(..., use_image=True)
    "image": """You receive an image of a photoluminescence spectrum (y vs x).
Return ONLY one JSON object. No explanations, no markdown.

Task:
- Identify up to max_peaks visible emission peaks.
- Estimate numeric values from axes.

Output JSON schema:
{
  "peaks": [
    { "center": number, "height": number, "fwhm": number|null, "prominence": number|null }
  ],
  "baseline": number|null
}

Rules:
- Use numeric estimates consistent with the plot axes.
- If baseline or FWHM are uncertain, set null.
- Do not exceed max_peaks.
- Output must be valid JSON.

You will receive an instruction text with max_peaks and a single image.
""",

    # Optional second pass to improve guesses before fitting
    "refine": """You receive a photoluminescence spectrum and a prior peak-guess JSON under "seed".
Return ONLY one JSON object matching the same schema. No explanations, no markdown.

Task:
- Improve initial guesses if inconsistent with data.
- Keep results usable as starting values for non-linear fitting.

Output JSON schema:
{
  "peaks": [
    { "center": number, "height": number, "fwhm": number|null, "prominence": number|null }
  ],
  "baseline": number|null
}

Guidelines:
- Remove spurious peaks; add a nearby secondary peak if a shoulder is evident (respect max_peaks).
- Keep centers within the x range; heights ≥ 0.
- Use null for uncertain baseline or FWHM.
- Output must be valid JSON.

Inputs follow under:
- "Series:" JSON (x, y, max_peaks, fields)
- "seed:" JSON (previous guesses)
- Optional "hints:" with residual regions or notes
""",

    # Strict structure guard
    "guardrails": """Return ONLY one JSON object that matches the requested schema.
No prose, no markdown, no code fences.
If impossible, return: {"peaks": [], "baseline": null}
""",

    # Optional: summarize low-quality fits and suggest revised seeds
    "plate_sweep": """You receive multiple spectra results with fit metrics.
Return ONLY one JSON object with revised seeds for any items with R2 < target_r2.

Input JSON (under "batch"):
[
  {
    "id": string,               // well/read identifier
    "x": number[],              // downsampled wavelengths
    "y": number[],              // downsampled intensities
    "seed": {                   // previous guesses (same schema as output)
      "peaks": [
        { "center": number, "height": number, "fwhm": number|null, "prominence": number|null }
      ],
      "baseline": number|null
    },
    "r2": number                // last fit R^2
  },
  ...
]

Constraints:
- Only include entries with r2 < target_r2 in the output.
- Keep centers within each entry's x range; heights ≥ 0.
- Use null for uncertain baseline or FWHM.
- Do not add text outside JSON.

Output JSON:
{
  "revised": {
    "<id>": {
      "peaks": [
        { "center": number, "height": number, "fwhm": number|null, "prominence": number|null }
      ],
      "baseline": number|null
    },
    ...
    }
}
""",

    "refine": """You are a peak fitting expert analyzing residuals to improve peak parameters.

Given current peak parameters and residual analysis, suggest refined peak positions, heights, and FWHM values.

Key principles:
1. Positive residual clusters suggest missing peaks or underestimated heights
2. Negative residual clusters suggest overestimated heights or wrong positions  
3. Systematic residual patterns indicate peak position/width adjustments needed
4. Focus on fixing systematic deviations, not random noise

Return ONLY JSON with keys 'peaks' (list) and 'baseline'. No explanations.

Example output:
{
  "peaks": [
    { "center": 650.5, "height": 1200.0, "fwhm": 25.0, "prominence": null },
    { "center": 720.2, "height": 800.0, "fwhm": 18.0, "prominence": null }
  ],
  "baseline": 50.0
}
""",
}

def get_prompt(name: str) -> str:
    """Fetch a system prompt by key ('numeric', 'image', 'refine', 'guardrails', 'plate_sweep')."""
    if name not in PROMPTS:
        raise KeyError(f"Unknown prompt: {name}. Available: {', '.join(sorted(PROMPTS))}")
    return PROMPTS[name]