import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Parámetros ---
fs = 48000          # frecuencia de muestreo [Hz]
N = 1024            # muestras (10 bins de separación entre 10k y 10.5k)
n = np.arange(N)

# Amplitudes lineales (dB → lineal: 0dB=1, -40dB=0.01, ruido -60dB=0.001)
A_sen  = 1.0                 # senoidal   0 dB  @ 10 kHz
A_cos  = 10**(-40/20)        # cosenoidal -40 dB @ 10.5 kHz
sigma  = 10**(-60/20)        # ruido      -60 dB

np.random.seed(42)
x = (A_sen  * np.sin(2*np.pi*10000*n/fs)
   + A_cos  * np.cos(2*np.pi*10500*n/fs)
   + sigma  * np.random.randn(N))

# --- Ventana Blackman ---
w = np.blackman(N)
U = np.mean(w**2)            # factor de corrección de potencia

# --- FFT con ventana ---
X = np.fft.rfft(x * w)
f = np.fft.rfftfreq(N, d=1/fs)

# Espectro en dB (magnitud al cuadrado / corrección)
S_dB = 20 * np.log10(np.abs(X) / (N * np.sqrt(U) / 2) + 1e-12)

# --- Bin del pico de la cosenoidal (10.5 kHz) ---
delta_f = fs / N             # resolución espectral
bin_10k  = int(round(10000 / delta_f))
bin_105k = int(round(10500 / delta_f))
sep_bins = bin_105k - bin_10k

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

ax.plot(f/1000, S_dB, color='#58a6ff', lw=0.9, label='Espectro (Blackman, N=1024)')

# Marcas de los picos
ax.axvline(10,   color='#f0883e', lw=1.2, ls='--', alpha=0.8)
ax.axvline(10.5, color='#3fb950', lw=1.2, ls='--', alpha=0.8)

ax.annotate('Senoidal\n0 dB @ 10 kHz',
            xy=(10, S_dB[bin_10k]), xytext=(7.5, -5),
            color='#f0883e', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#f0883e', lw=1))

ax.annotate(f'Cosenoidal\n−40 dB @ 10.5 kHz\n(bin {bin_105k}, sep={sep_bins} bins)',
            xy=(10.5, S_dB[bin_105k]), xytext=(12, -30),
            color='#3fb950', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#3fb950', lw=1))

# Líneas de referencia de nivel
for nivel, label, col in [(-40, '−40 dB (cosenoidal)', '#f85149'),
                           (-60, '−60 dB (piso ruido)', '#8b949e')]:
    ax.axhline(nivel, color=col, lw=0.8, ls=':', alpha=0.7)
    ax.text(f[-1]/1000 * 0.98, nivel + 1, label,
            color=col, fontsize=8, ha='right')

# Anotación separación en bins
ax.annotate('', xy=(10.5, -55), xytext=(10, -55),
            arrowprops=dict(arrowstyle='<->', color='white', lw=1))
ax.text(10.25, -53, f'{sep_bins} bins', color='white',
        fontsize=8, ha='center')

# Info ventana
info = (f'Ventana: Blackman   N={N}   fs={fs//1000} kHz\n'
        f'Δf = {delta_f:.1f} Hz/bin   Lóbulo lateral ≈ −74 dB')
ax.text(0.01, 0.97, info, transform=ax.transAxes,
        color='#c9d1d9', fontsize=8.5, va='top',
        bbox=dict(facecolor='#161b22', edgecolor='#30363d', boxstyle='round,pad=0.4'))

ax.set_xlim(0, fs/2000)
ax.set_ylim(-90, 10)
ax.set_xlabel('Frecuencia [kHz]', color='#c9d1d9')
ax.set_ylabel('Magnitud [dB]', color='#c9d1d9')
ax.set_title('Ejercicio 4 — Espectro de señal compuesta con ventana Blackman',
             color='#e6edf3', fontsize=11, pad=12)
ax.tick_params(colors='#8b949e')
for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')
ax.grid(True, color='#21262d', lw=0.6)

# Zoom panel — región de interés 9–12 kHz
axins = ax.inset_axes([0.55, 0.45, 0.42, 0.45])
mask = (f >= 9000) & (f <= 12000)
axins.plot(f[mask]/1000, S_dB[mask], color='#58a6ff', lw=1)
axins.axvline(10,   color='#f0883e', lw=1, ls='--', alpha=0.8)
axins.axvline(10.5, color='#3fb950', lw=1, ls='--', alpha=0.8)
axins.axhline(-40, color='#f85149', lw=0.8, ls=':')
axins.axhline(-60, color='#8b949e', lw=0.8, ls=':')
axins.set_facecolor('#161b22')
axins.tick_params(colors='#8b949e', labelsize=7)
axins.set_title('Zoom 9–12 kHz', color='#c9d1d9', fontsize=8)
axins.grid(True, color='#21262d', lw=0.5)
for spine in axins.spines.values():
    spine.set_edgecolor('#30363d')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/espectro_ejercicio4.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Guardado: espectro_ejercicio4.png")
