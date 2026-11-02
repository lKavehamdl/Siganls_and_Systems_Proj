# Spectrogram Analysis Tool

Final project for the **Signals & Systems** course: a desktop app that computes and shows the
**Short-Time Fourier Transform (STFT)** spectrogram of synthetic test signals or WAV files.
The FFT, the window functions and the STFT are written from scratch with NumPy. No
`np.fft.fft` / `scipy.signal.stft` is used for the transform itself.

**Author:** Kaveh Ahmadi (40131904)

---

## Features

- **Recursive radix-2 Cooley–Tukey FFT** (`compute_fft`), O(N log N), with automatic zero padding
  up to the next power of two
- **Window functions** (`create_window`):
  - Hann: `w(n) = 0.5 · (1 − cos(2πn / (N−1)))`
  - Hamming: `w(n) = 0.54 − 0.46 · cos(2πn / (N−1))`
  - Rectangular: `w(n) = 1`
- **STFT** (`calculate_stft`) with configurable window size, overlap and window type
- **Time/frequency axes** (`compute_axes`): each frame's time is taken at the window centre, and
  frequency bins are spaced `fs / N` apart
- **Built-in test signals** (6 s at 8 kHz):
  | Signal | Definition |
  |---|---|
  | Linear chirp | `sin(2π(f0·t + k/2·t²))` |
  | Exponential chirp | `sin(2π·f0·(rᵗ − 1)/ln r)` |
  | Piecewise sine | 300 Hz → 500 Hz → 400 Hz (2 s each) |
  | Piecewise sine combination | (300+700) → (500+900) → (400+1200) Hz (2 s each) |
- **WAV import**: handles 8/16/32-bit PCM, mixes stereo down to mono and keeps only the first 10 s
- **Tkinter GUI** with an embedded Matplotlib time-domain plot and a dB-scaled spectrogram
  (positive frequencies only)

## Requirements

- Python 3.8+
- `numpy`, `scipy`, `matplotlib`
- `tkinter` (part of most Python installs; on Arch Linux install `tk`, on Debian/Ubuntu `python3-tk`)

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib
```

## Usage

```bash
python Signal_PRJ_40131904.py
```

1. Pick a **Signal** type. For the chirps, set **f0** (start frequency, Hz) and **k/r**: the
   chirp rate `k` (Hz/s) for the linear chirp, or the growth factor `r` (> 1) for the exponential one.
2. Click **Create Signal**, or click **Import WAV** to load a file
   (for example `file_example_WAV_1MG.wav`).
3. Set **Window Size** (default 512), **Overlap** in samples (default 256) and **Window Type**.
4. Click **Analyze STFT** to draw the spectrogram.

Tips:
- A power-of-two window size avoids extra zero padding inside each frame.
- The overlap must be smaller than the window size.
- A bigger window gives finer frequency resolution and coarser time resolution, and a smaller
  window does the opposite.

## Project Structure

```
.
├── Signal_PRJ_40131904.py   # GUI application + FFT / window / STFT implementation
├── report.pdf               # Project report (compiled)
├── report.tex               # Project report source (Persian, XeLaTeX)
├── file_example_WAV_1MG.wav # Sample WAV file for testing
├── PRJ_40131904.zip         # Submission archive (report.pdf + source)
└── README.md
```

## Report

The compiled report is in [`report.pdf`](report.pdf). Its source, `report.tex`, is written in Persian. It covers the window functions, the
Cooley–Tukey FFT derivation, zero padding, the STFT formulation, how the axes are computed, and
the test signals. It uses `xepersian`, so it has to be built with **XeLaTeX** and needs the
*Nazanin* and *Times New Roman* fonts. It also needs `images/AUTlogo.png` (the `images/`
folder is not tracked by git).

```bash
xelatex report.tex
```
