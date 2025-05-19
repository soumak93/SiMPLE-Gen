# SiMPLE-Gen
Simulated Mock Population of Lyman-Alpha Emitters Generator

A modular, end-to-end pipeline to generate simulated Lyman-α emitters from cosmological simulation outputs.


---

## ⚙️ Setup

1. **Clone** the repo:

   ```bash
   git clone https://github.com/yourorg/SiMPLE-Gen.git
   cd SiMPLE-Gen
   ```

2. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare** your raw data:

   * Ensure the necessary directories exist by running:

     ```bash
     mkdir -p data/raw data/processed
     ```
   * Place simulation files (density, temperature, ionization, velocity, halo positions & masses) in `data/raw/`.
   * Edit `SiMPLE-Gen/config.py ▶︎ INPUT_DIR` to point at that folder.
  
---

## 📦 Repository Structure

```
SiMPLE-Gen/              # root folder (repo name)
├── run.py               # master pipeline script
├── requirements.txt     # Python dependencies
├── README.md            # this file
├── .gitignore
└── SiMPLE-Gen/          # Python package
    ├── config.py        # all RAW & PATHS definitions
    ├── gen.py           # Step 1: sightline generation
    ├── spec.py          # Step 2: τ(z) calculation
    ├── abundance.py     # Step 3: abundance matching
    ├── assign.py        # Step 4: LLya & REW sampling
    └── damping.py       # Step 5: damping transmission
```

Simulation inputs live outside this repo under `data/raw/`, and outputs are written to `data/processed/` by default.


---

## 🚀 Running the Pipeline

Execute the master script:

```bash
python run.py
```

This calls each step in sequence: `gen.py`, `spec.py`, `abundance.py`, `assign.py`, `damping.py`.

---

## 🛠️ Module Descriptions

* **`gen.py`**: Extracts 1D neutral-hydrogen, temperature, and velocity sightlines around halos.
* **`spec.py`**: Computes Lyman-α optical depth τ(z) along each sightline using your Voigt-profile routine.
* **`abundance.py`**: Builds a Sheth–Tormen halo mass function, Schechter UV LF, applies duty cycle, then matches halo masses ↔ UV magnitudes.
* **`assign.py`**: Defines probability distributions for rest-frame equivalent width (REW) and Lyman-α luminosity, and Monte Carlo-samples each halo’s values.
* **`damping.py`**: Convolves τ(v) with a halo-velocity Gaussian to compute the equivalent-width decrease ratio.

All file paths are centralized in **`config.py`**—no hidden hardcoded paths.

---

