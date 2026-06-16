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