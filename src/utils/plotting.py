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