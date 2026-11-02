import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.io import wavfile
from matplotlib.figure import Figure

def generate_linear_chirp(f_start: float, rate: float, time: float) -> float:
    phase = 2 * math.pi * (f_start * time + 0.5 * rate * time**2)
    return math.sin(phase)
vec_linear_chirp = np.vectorize(generate_linear_chirp)

def generate_exp_chirp(f_initial: float, growth: float, time: float) -> float:
    if growth <= 1:
        growth = 1.0001  # Small positive value to avoid invalid logarithm
    phase = 2 * math.pi * f_initial * ((growth**time - 1) / math.log(growth))
    return math.sin(phase)
vec_exp_chirp = np.vectorize(generate_exp_chirp)

def piecewise_sine_wave(time: float) -> float:
    if 0 <= time < 2:
        return math.sin(2 * math.pi * 300 * time)
    elif 2 <= time < 4:
        return math.sin(2 * math.pi * 500 * time)
    elif 4 <= time < 6:
        return math.sin(2 * math.pi * 400 * time)
    return 0.0
vec_piecewise_sine = np.vectorize(piecewise_sine_wave)

def piecewise_sine_sum(time: float) -> float:
    if 0 <= time < 2:
        return math.sin(2 * math.pi * 300 * time) + math.sin(2 * math.pi * 700 * time)
    elif 2 <= time < 4:
        return math.sin(2 * math.pi * 500 * time) + math.sin(2 * math.pi * 900 * time)
    elif 4 <= time < 6:
        return math.sin(2 * math.pi * 400 * time) + math.sin(2 * math.pi * 1200 * time)
    return 0.0
vec_piecewise_sine_sum = np.vectorize(piecewise_sine_sum)

def compute_fft(signal):
    """Fast Fourier Transform using Cooley-Tukey approach"""
    length = len(signal)
    if length <= 1:
        return signal
    if length & (length - 1) != 0:  # Check if not power of 2
        next_power = 2 ** math.ceil(math.log2(length))
        signal = np.pad(signal, (0, next_power - length), mode='constant')
        length = next_power
    even_terms = compute_fft(signal[::2])
    odd_terms = compute_fft(signal[1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(length // 2) / length)
    return np.concatenate([even_terms + twiddle * odd_terms, even_terms - twiddle * odd_terms])

def create_window(window_kind, size):
    if window_kind.lower() == 'hann':
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(size) / (size - 1)))
    elif window_kind.lower() == 'hamming':
        return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    elif window_kind.lower() == 'rectangular':
        return np.ones(size)
    else:
        raise ValueError(f"Invalid window kind: {window_kind}. Use 'hann', 'hamming', or 'rectangular'.")

def calculate_stft(signal, win_size, overlap_samples, window_kind='hann'):
    signal = np.asarray(signal)
    step_size = win_size - overlap_samples
    signal_len = len(signal)
    frame_count = max(1, (signal_len - win_size) // step_size + 1)
    window = create_window(window_kind, win_size)
    stft_output = np.zeros((win_size, frame_count), dtype=complex)
    
    for frame in range(frame_count):
        start = frame * step_size
        end = start + win_size
        if end <= signal_len:
            frame_data = signal[start:end] * window
        else:
            frame_data = np.zeros(win_size)
            frame_data[:signal_len - start] = signal[start:] * window[:signal_len - start]
        stft_output[:, frame] = compute_fft(frame_data)
    return stft_output

def compute_axes(signal_len, sample_freq, win_size, overlap_samples):
    step_size = win_size - overlap_samples
    frame_count = max(1, (signal_len - win_size) // step_size + 1)
    time_points = np.arange(frame_count) * step_size + win_size // 2
    time_points = time_points / sample_freq
    freq_points = np.fft.fftfreq(win_size, 1/sample_freq)
    return time_points, freq_points

class SpectrogramApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Spectrogram Analysis Tool")
        self.window.geometry("1200x800")
        
        self.signal_data = None
        self.sampling_rate = 8000
        self.signal_duration = 6.0
        
        self.setup_ui()
        self.create_initial_signal()
    
    def setup_ui(self):
        # Main container
        container = ttk.Frame(self.window)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Controls section
        controls = ttk.LabelFrame(container, text="Settings", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))
        
        # Signal type selector
        ttk.Label(controls, text="Signal:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.signal_choice = tk.StringVar(value="Linear Chirp")
        signal_selector = ttk.Combobox(controls, textvariable=self.signal_choice, 
                                     values=["Linear Chirp", "Exponential Chirp", "Piecewise Sine", 
                                             "Piecewise Sine Combination"])
        signal_selector.grid(row=0, column=1, padx=(0, 20))
        signal_selector.bind('<<ComboboxSelected>>', self.update_signal_type)
        
        # Parameters section
        params = ttk.Frame(controls)
        params.grid(row=0, column=2, padx=20)
        
        ttk.Label(params, text="f0:").grid(row=0, column=0)
        self.f0_input = tk.StringVar(value="100")
        ttk.Entry(params, textvariable=self.f0_input, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(params, text="k/r:").grid(row=0, column=2)
        self.rate_input = tk.StringVar(value="50")
        ttk.Entry(params, textvariable=self.rate_input, width=8).grid(row=0, column=3, padx=2)
        
        # STFT settings
        stft_settings = ttk.Frame(controls)
        stft_settings.grid(row=1, column=0, columnspan=4, pady=10)
        
        ttk.Label(stft_settings, text="Window Size:").grid(row=0, column=0)
        self.win_size_input = tk.StringVar(value="512")
        ttk.Entry(stft_settings, textvariable=self.win_size_input, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(stft_settings, text="Overlap:").grid(row=0, column=2)
        self.overlap_input = tk.StringVar(value="256")
        ttk.Entry(stft_settings, textvariable=self.overlap_input, width=8).grid(row=0, column=3, padx=5)
        
        ttk.Label(stft_settings, text="Window Type:").grid(row=0, column=4)
        self.window_kind_input = tk.StringVar(value="hann")
        ttk.Combobox(stft_settings, textvariable=self.window_kind_input, 
                    values=["hann", "hamming", "rectangular"], width=10).grid(row=0, column=5, padx=5)
        
        # Action buttons
        actions = ttk.Frame(controls)
        actions.grid(row=2, column=0, columnspan=4, pady=10)
        
        ttk.Button(actions, text="Create Signal", command=self.create_signal).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions, text="Import WAV", command=self.import_wav).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions, text="Analyze STFT", command=self.analyze_stft).pack(side=tk.LEFT, padx=5)
        
        # Plot area
        self.setup_plots(container)
    
    def setup_plots(self, parent):
        plot_area = ttk.Frame(parent)
        plot_area.pack(fill=tk.BOTH, expand=True)
        
        self.figure = Figure(figsize=(12, 8))
        self.time_plot = self.figure.add_subplot(2, 1, 1)
        self.freq_plot = self.figure.add_subplot(2, 1, 2)
        
        self.plot_canvas = FigureCanvasTkAgg(self.figure, plot_area)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_initial_signal(self):
        self.create_signal()
    
    def create_signal(self):
        try:
            signal_type = self.signal_choice.get()
            time_array = np.linspace(0, self.signal_duration, int(self.sampling_rate * self.signal_duration))
            
            if signal_type == "Linear Chirp":
                f0 = float(self.f0_input.get())
                rate = float(self.rate_input.get())
                self.signal_data = vec_linear_chirp(f0, rate, time_array)
            elif signal_type == "Exponential Chirp":
                f0 = float(self.f0_input.get())
                rate = float(self.rate_input.get())
                if rate <= 1:
                    rate = 1.1
                self.signal_data = vec_exp_chirp(f0, rate, time_array)
            elif signal_type == "Piecewise Sine":
                self.signal_data = vec_piecewise_sine(time_array)
            elif signal_type == "Piecewise Sine Combination":
                self.signal_data = vec_piecewise_sine_sum(time_array)
            
            self.time_plot.clear()
            self.time_plot.plot(time_array, self.signal_data)
            self.time_plot.set_title(f"{signal_type} - Time Domain")
            self.time_plot.set_xlabel("Time (s)")
            self.time_plot.set_ylabel("Amplitude")
            self.time_plot.grid(True)
            
            self.freq_plot.clear()
            self.freq_plot.set_title("Spectrogram (Click 'Analyze STFT' to generate)")
            
            self.plot_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create signal: {str(e)}")
    
    def import_wav(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Select WAV File",
                filetypes=[("WAV files", "*.wav")]
            )
            
            if file_path:
                sample_rate, signal = wavfile.read(file_path)
                
                if signal.dtype == np.int16:
                    signal = signal.astype(np.float32) / 32768.0
                elif signal.dtype == np.int32:
                    signal = signal.astype(np.float32) / 2147483648.0
                elif signal.dtype == np.uint8:
                    signal = (signal.astype(np.float32) - 128) / 128.0
                
                if len(signal.shape) > 1:
                    signal = np.mean(signal, axis=1)
                
                max_samples = sample_rate * 10
                if len(signal) > max_samples:
                    signal = signal[:max_samples]
                
                self.signal_data = signal
                self.sampling_rate = sample_rate
                self.signal_duration = len(signal) / sample_rate
                
                time_array = np.linspace(0, self.signal_duration, len(signal))
                self.time_plot.clear()
                self.time_plot.plot(time_array, signal)
                self.time_plot.set_title("Imported WAV - Time Domain")
                self.time_plot.set_xlabel("Time (s)")
                self.time_plot.set_ylabel("Amplitude")
                self.time_plot.grid(True)
                
                self.freq_plot.clear()
                self.freq_plot.set_title("Spectrogram (Click 'Analyze STFT' to generate)")
                
                self.plot_canvas.draw()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import WAV: {str(e)}")
    
    def analyze_stft(self):
        if self.signal_data is None:
            messagebox.showwarning("Warning", "No signal loaded. Create or import a signal first.")
            return
    
        try:
            win_size = int(self.win_size_input.get())
            overlap_samples = int(self.overlap_input.get())
            window_kind = self.window_kind_input.get()
            
            # Compute STFT
            stft_data = calculate_stft(self.signal_data, win_size, overlap_samples, window_kind)
            time_axis, freq_axis = compute_axes(len(self.signal_data), self.sampling_rate, 
                                              win_size, overlap_samples)
            
            # Compute magnitude and convert to dB
            magnitude = np.abs(stft_data)
            # Clip magnitude to avoid log(0) or invalid values
            magnitude = np.clip(magnitude, 1e-10, None)
            magnitude_db = 20 * np.log10(magnitude)
            
            # Check for invalid values
            if np.any(np.isnan(magnitude_db)) or np.any(np.isinf(magnitude_db)):
                raise ValueError("STFT computation produced invalid values (NaN or Inf).")
            
            # Plot spectrogram (only positive frequencies)
            self.freq_plot.clear()
            freq_positive = freq_axis[:len(freq_axis)//2]
            magnitude_positive = magnitude_db[:len(freq_axis)//2, :]
            
            im = self.freq_plot.imshow(magnitude_positive, aspect='auto', origin='lower',
                                     extent=[time_axis[0], time_axis[-1], freq_positive[0], freq_positive[-1]],
                                     cmap='viridis')
            
            self.freq_plot.set_title("STFT Spectrogram")
            self.freq_plot.set_xlabel("Time (s)")
            self.freq_plot.set_ylabel("Frequency (Hz)")
            
            # Safely remove existing colorbar
            if hasattr(self, 'colorbar') and self.colorbar is not None:
                try:
                    self.colorbar.remove()
                except Exception as e:
                    print(f"Warning: Failed to remove colorbar: {e}")
            self.colorbar = self.figure.colorbar(im, ax=self.freq_plot, label='Magnitude (dB)')
            
            self.plot_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute STFT: {str(e)}")
        
            
    def update_signal_type(self, event=None):
        signal_type = self.signal_choice.get()
        # No changes needed for parameter inputs in this version
        pass

def run_app():
    root = tk.Tk()
    app = SpectrogramApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_app()