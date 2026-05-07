import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    """
    Genera una señal senoidal.
    """
    ts = 1/fs
    tt = np.arange(nn) * ts
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    return tt, xx

# --- 1. Parámetros de la simulación ---
nn = 500          # Número de muestras (N)
fs = 500          # Frecuencia de muestreo
df = fs / nn      # Resolución espectral (Delta f = 1 Hz)
vmax = np.sqrt(2) # Amplitud para tener Potencia Unitaria (A^2 / 2 = 1)
dc = 0
ph = 0

# Frecuencias solicitadas (k0 * df)
ks = [nn/4, nn/4 + 0.25, nn/4 + 0.5]
titulos = ['Bin exacto (k0=125)', 'Desintonía leve (k0=125.25)', 'Máxima fuga (k0=125.5)']

# Configuración de la figura
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

for i, k0 in enumerate(ks):
    f0 = k0 * df
    tt, s = mi_funcion_sen(vmax, dc, f0, ph, nn, fs)
    
    # --- 2. Identidad de Parseval (Punto b) ---
    # Potencia en el tiempo
    pot_tiempo = np.mean(s**2)
    
    # FFT sin padding
    # Dividimos por N para que la magnitud sea independiente del largo de la señal
    XX_s = np.fft.fft(s) / nn 
    # Potencia en frecuencia (Suma de los cuadrados de los bins)
    pot_frec = np.sum(np.abs(XX_s)**2)
    
    # Espectro unilateral para graficar (multiplicamos por 2 para conservar energía)
    psd_s = np.abs(XX_s)**2
    psd_s[1:nn//2] *= 2 
    frec_s = np.fft.fftfreq(nn, 1/fs)[:nn//2]
    
    # --- 3. Zero Padding (Punto c) ---
    # Agregamos 9*N ceros al final (Total 10*N muestras)
    cant_padds = 9
    sf = np.pad(s, (0, nn * cant_padds), 'constant')
    N_total = len(sf)
    
    # FFT con padding (Normalizamos por el N original para comparar amplitudes)
    XX_sf = np.fft.fft(sf) / nn
    psd_sf = np.abs(XX_sf)**2
    psd_sf[1:N_total//2] *= 2
    frec_sf = np.fft.fftfreq(N_total, 1/fs)[:N_total//2]

    # --- 4. Gráficos (Punto a) ---
    # Usamos escala Logarítmica (dB) para ver mejor las faldas del espectro
    axes[i].plot(frec_sf, 10 * np.log10(psd_sf[:N_total//2] + 1e-12), 
                 color='royalblue', label='Zero Padding (Interp. Continua)')
    axes[i].plot(frec_s, 10 * np.log10(psd_s[:nn//2] + 1e-12), 
                 'ro', markersize=5, label='Bins DFT Original')
    
    axes[i].set_title(f"{titulos[i]}\nPotencia T: {pot_tiempo:.4f} | Potencia F: {pot_frec:.4f}")
    axes[i].set_ylabel("Magnitud [dB]")
    axes[i].set_xlim(110, 140) # Hacemos zoom en la zona de interés
    axes[i].set_ylim(-60, 5)
    axes[i].grid(True, linestyle=':', alpha=0.6)
    if i == 0:
        axes[i].legend()

axes[2].set_xlabel("Frecuencia [Hz]")
plt.tight_layout()
plt.show()