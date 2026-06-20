import matplotlib.pyplot as plt
import numpy as np

def plot_spectral_gc(
        frequencies,
        gc,
        title=""
):
#     plt.figure(figsize=(8, 4))

#     plt.plot(frequencies, gc)
#     plt.xlabel("Frequency (rad/sample)")
#     plt.ylabel("GC")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig('results/'+title.replace('->', '_to_')+'.png')

        plt.figure(figsize=(9,4))
        plt.plot(frequencies/np.pi, gc, linewidth=2)
        plt.xlabel("Normalized Frequency (x pi rad/sample)")
        plt.ylabel("Spectral GC")
        plt.grid(True)
        plt.title(title)
        plt.tight_layout()
        plt.savefig('results/'+title.replace('->', '_to_')+'.png')

def plot_residual_acf(acf_values, title=""):
        plt.figure(figsize=(8,4))
        plt.stem(range(1, len(acf_values)+1), acf_values)
        plt.axhline(0, linestyle="--")
        plt.title(title)
        plt.xlabel("Lag")
        plt.ylabel("ACF")
        plt.tight_layout()
        plt.savefig('results/'+title+'.png')

def plot_information_criteria(results):
        orders = [ r["order"] for r in results ]

        aic = [ r["aic"] for r in results ]
        bic = [ r["bic"] for r in results ]

        plt.figure(figsize=(8, 4))
        plt.plot(orders, aic, label="AIC", marker="o")
        plt.plot(orders, bic, label="BIC", marker="o")
        plt.xlabel("VAR Order")
        plt.ylabel("Criterion Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/Order vs. Information Criteria.png")

def plot_power_spectrum(frequencies, spectrum, title=""):
        plt.figure(figsize=(9,4))
        plt.plot(frequencies / np.pi, spectrum, linewidth=2)
        plt.xlabel("Normalized Frequency (xpi)")
        plt.ylabel("Power")
        plt.grid(True)
        plt.title(title)
        plt.tight_layout()
        plt.savefig('results/'+title+'.png')

def plot_gc_evolution(times, gc_values, change_point=None, title=""):
        plt.figure(figsize=(10, 4))
        plt.plot(times, gc_values, linewidth=2)

        if change_point is not None:
                plt.axvline(change_point, linestyle="--")
        
        plt.xlabel("Time")
        plt.ylabel("GC")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/'+title+'.png')

def plot_true_vs_estimated_gc(times, true_coupling, gc_times, gc_values):
        plt.figure(figsize=(10, 5))
        plt.plot(true_coupling, label="True Coupling")
        plt.plot(gc_times, gc_values, label="Sliding GC")
        plt.xlabel("Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/True vs Estimated GC.png')

def plot_adaptive_coupling(true_coupling, estimated_coupling, order):
        plt.figure(figsize=(10, 5))
        plt.plot(true_coupling, label="True")
        plt.plot(estimated_coupling, label="RLS Estimate")
        plt.xlabel("Time")
        plt.ylabel("Coupling")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/Adaptive coupling with order '+str(order)+'.png')

def plot_parameter_convergence(history, row, col, title=""):
        values = history[:, row, col]
        plt.figure(figsize=(8, 4))
        plt.plot(values)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/'+title+'.png')

def plot_gc_surface(gc_surface, frequencies):
        plt.figure(figsize=(10, 6))
        plt.imshow(
                gc_surface.T, 
                aspect="auto", 
                origin="lower",
                extent=[0, gc_surface.shape[0], frequencies[0]/np.pi, frequencies[-1]/np.pi])
        plt.colorbar(label="GC")
        plt.xlabel("Time")
        plt.ylabel("Frequency (x pi)")
        plt.title("Adaptive Spectral GC")
        plt.tight_layout()
        plt.savefig('results/Adaptive Spectral GC.png')

def plot_spectrum_comparison(
        frequencies,
        spectrum_1,
        spectrum_2,
        label_1="Spectrum 1",
        label_2="Spectrum 2",
        title=""
):
        spectrum_1 = np.asarray(spectrum_1)
        spectrum_2 = np.asarray(spectrum_2)

        # Normalize both spectra for shape comparison.
        # Raw amplitudes may differ because parametric and multitaper
        # estimators use different scaling conventions.
        spectrum_1_norm = spectrum_1 / np.max(spectrum_1)
        spectrum_2_norm = spectrum_2 / np.max(spectrum_2)

        plt.figure(figsize=(9, 4))

        plt.plot(
                frequencies / np.pi,
                spectrum_1_norm,
                linewidth=2,
                label=label_1
        )

        plt.plot(
                frequencies / np.pi,
                spectrum_2_norm,
                linewidth=1,
                label=label_2
        )

        plt.xlabel("Normalized frequency (x pi)")
        plt.ylabel("Normalized power")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/'+title+'.png')

def plot_gc_comparison(
                frequencies,
                gc_1,
                gc_2,
                label_1="GC 1",
                label_2="GC 2",
                title=""
):
        gc_1 = np.asarray(gc_1)
        gc_2 = np.asarray(gc_2)

        plt.figure(figsize=(9, 4))
        plt.plot(frequencies/np.pi, gc_1, linewidth=2, label=label_1)
        plt.plot(frequencies/np.pi, gc_2, linewidth=2, label=label_2)
        plt.xlabel("Normalized frequency (x pi)")
        plt.ylabel("Spectral GC")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/'+title+'.png')