import numpy as np
from typing import Callable, Dict
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from ising_model_physics import IsingModel

class BasisLogger: 
    def __init__(self, calls: Dict[str, Callable], start_iteration = 0, sample_rate = 1): 
        self.calls = calls
        self.log_frame = pd.DataFrame(columns = calls.keys()) # Dataframe, each column has an observable, the observables are encoded in the calls
        self.start_iteration = start_iteration
        self.sample_rate = sample_rate
        self.iteration = 0

    def __call__(self, obj):
        self.iteration += 1
        if self.iteration > self.start_iteration and self.iteration % self.sample_rate == 0: 
            values = {name: func(obj) for name, func in self.calls.items()}
            self.log_frame.loc[len(self.log_frame)] = values

    def __getitem__(self, key): 
        return np.array(self.log_frame[key])
    
    def mean(self, key):
        return np.mean(self[key])

    def std(self, key):
        return np.std(self[key])
    
class AutocorrLogger(BasisLogger):
    def __init__(self, calls, start_iteration=0, sample_rate=1):
        super().__init__(calls, start_iteration, sample_rate)

    def _lag_steps(self, t):
        return t // self.sample_rate

    def calculate_autocorrelation(self, logged_observable, t):
        data = np.asarray(self[logged_observable]).flatten()
        n = len(data)

        t_logging_interval = self._lag_steps(t)
        if t_logging_interval >= n:
            # Automatically skip lags that are too long
            return np.nan

        corr = np.mean(data[:n - t_logging_interval] * data[t_logging_interval:])
        return corr

    def generate_default_t_samples(self, nr_samples=50):
        max_lag = len(self.log_frame) - 1
        nr_samples = min(nr_samples, max_lag)
        return np.arange(1, nr_samples + 1) * self.sample_rate

    def calculate_autocorrelations(self, logged_observable, t_samples=None):
        if t_samples is None:
            t_samples = self.generate_default_t_samples()

        t_samples = np.asarray(t_samples).flatten()
        autocorrs = np.array([self.calculate_autocorrelation(logged_observable, t) for t in t_samples])
        # Remove NaNs automatically
        valid_mask = ~np.isnan(autocorrs)
        return autocorrs[valid_mask], t_samples[valid_mask]

    def plot_autocorrelations(self, logged_observable, t_samples=None, autocorrelations=None):
        if t_samples is None or autocorrelations is None:
            autocorrelations, t_samples = self.calculate_autocorrelations(logged_observable, t_samples)

        t_samples = np.asarray(t_samples).flatten()
        autocorrelations = np.asarray(autocorrelations).flatten()
        mask = autocorrelations > 0

        if np.sum(mask) < 2:
            print("Not enough valid autocorrelation points to fit.")
            plt.plot(t_samples, autocorrelations, "o")
            plt.show()
            return

        slope, intercept, *_ = stats.linregress(t_samples[mask], np.log(autocorrelations[mask]))
        tau = -1 / slope

        plt.title("Autocorrelation")
        plt.xlabel("Time lag [steps]")
        plt.ylabel("Autocorrelation [unit²]")
        plt.plot(t_samples, autocorrelations, "o", label="Measured")
        t_cont = np.linspace(min(t_samples), max(t_samples), 200)
        plt.plot(t_cont, np.exp(intercept + slope * t_cont), label=fr"$e^{{-t/\tau}}, \tau = {tau:.2f}$")
        plt.legend()
        plt.grid(True)
        plt.show()



class IsingModelLogger(AutocorrLogger): 
    MODES = {
        "default": {
            "t": IsingModel.get_steps,
            "E": IsingModel.hamiltonian,
            "s": IsingModel.get_lattice,
            "magnetisation": IsingModel.magnetization,
            "size": IsingModel.get_size,
        },
        #"swendsen": {
        #    "susceptibility": SwendsenWang.get_susceptibility,
        #},
        #"wolff": {
        #    "susceptibility": Wolff.get_susceptibility,
        #},
    }

    def __init__(self, mode="default", extra_calls=None, start_iteration=0, sample_rate=1):
        calls = dict(self.MODES.get("default", {}))
        if mode in self.MODES and mode != "default":
            calls.update(self.MODES[mode])
        if extra_calls:
            calls.update(extra_calls)
        super().__init__(calls, start_iteration, sample_rate)

    def get_energy(self):
        
        if "E" not in self.log_frame:
            print("empty energy")
            return None

        E = np.array(self["E"], dtype=float)
        mean_E = np.mean(E)
        std_E = np.std(E, ddof=1) 
        stderr_E = std_E / np.sqrt(len(E))  

        return {"mean": mean_E, "std": std_E, "stderr": stderr_E}


    def plot_energy(self):
        
        if "E" in self.log_frame and "t" in self.log_frame:
            plt.title("Energy vs Time")
            plt.plot(self["t"], self["E"], label="Energy")
            plt.xlabel("Steps")
            plt.ylabel("Energy")
            plt.legend()
            plt.show()


    def get_susceptibility(self):
        
        if "susceptibility" in self.log_frame:
            s = np.array(self["susceptibility"], dtype=float)
            mean_s = np.mean(s)
            std_s = np.std(s, ddof=1)
            stderr_s = std_s / np.sqrt(len(s))
            return {"mean": mean_s, "std": std_s, "stderr": stderr_s}

        elif "magnetisation" in self.log_frame and "size" in self.log_frame:
            m = np.array(self["magnetisation"], dtype=float)
            size = float(self["size"][0])
            chi = (np.mean(m ** 2) - np.mean(m) ** 2) / size
            err = np.sqrt(2 * chi ** 2 / (len(m) - 1))
            return {"mean": chi, "stderr": err}

        return None
