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

# %% Parámetros
vmax = np.sqrt(2)
dc = 0 
ph = 0 
nn = 500 
fs = 500 
df = fs / nn

# Frecuencias 
f0 = (nn/4) * df         # 125.0 Hz (Bin exacto)
f1 = (nn/4 + 0.25) * df  # 125.25 Hz
f2 = (nn/4 + 0.5) * df   # 125.5 Hz (Máxima fuga)

#%%
# Generacion de senales 
tt, s0 = mi_funcion_sen(vmax, dc, f0, ph, nn, fs)
tt, s1 = mi_funcion_sen(vmax, dc, f1, ph, nn, fs)
tt, s2 = mi_funcion_sen(vmax, dc, f2, ph, nn, fs)

# ---  Identidad de Parseval (Punto b) ---
# Potencia en el tiempo
pot_tiempo0 = np.mean(s0**2)
pot_tiempo1 = np.mean(s1**2)
pot_tiempo2 = np.mean(s2**2)
    
# FFT sin padding
# Dividimos por N para que la magnitud sea independiente del largo de la señal
XX_s0 = np.fft.fft(s0) / nn
XX_s1 = np.fft.fft(s1) / nn
XX_s2 = np.fft.fft(s2) / nn   
# Potencia en frecuencia (Suma de los cuadrados de los bins)
pot_frec0 = np.sum(np.abs(XX_s0)**2)
pot_frec1 = np.sum(np.abs(XX_s1)**2)
pot_frec2 = np.sum(np.abs(XX_s2)**2)
    
# Espectro unilateral para graficar (multiplicamos por 2 para conservar energía)
psd_s0 = np.abs(XX_s0)**2
psd_s0[1:nn//2] *= 2
frec_s = np.arange(0, fs/2, df)


psd_s1 = np.abs(XX_s1)**2
psd_s1[1:nn//2] *= 2 


psd_s2 = np.abs(XX_s2)**2
psd_s2[1:nn//2] *= 2 

# %% Generación de gráficos 

# Lista de datos para iterar y generar gráficos separados
datos = [
    (s0, XX_s0, f0, pot_tiempo0, pot_frec0, "Bin exacto"),
    (s1, XX_s1, f1, pot_tiempo1, pot_frec1, "Desintonía leve"),
    (s2, XX_s2, f2, pot_tiempo2, pot_frec2, "Máxima fuga")
]

for s, XX, freq, p_t, p_f, nombre in datos:
    plt.figure(figsize=(10, 4))
    
    # 2. Bins originales de la DFT
    # Normalización consistente para los puntos rojos
    psd_puntos = np.abs(XX[:nn//2] / (1/2))
    plt.plot(frec_s, 20 * np.log10(psd_puntos + 1e-12), 'ro', markersize=5, label='Bins DFT')
    
    # Títulos y formato
    plt.suptitle("Densidades Espectrales de Potencia (PDS's)", fontsize=14)
    plt.title(f"{nombre} (k0={freq/df:.2f}) | Potencia T: {p_t:.4f} F: {p_f:.4f}", fontsize=10)
    
    plt.xlim([110, 140])
    plt.ylabel("Magnitud [dB]")
    plt.xlabel("Frecuencia [Hz]")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

#%%

# técnica de zero padding

cant_padds = 9


sp0 = np.concatenate((s0, np.zeros(nn * cant_padds)))
sp1 = np.concatenate((s1, np.zeros(nn * cant_padds)))
sp2 = np.concatenate((s2, np.zeros(nn * cant_padds)))
N_total = len(sp0)

# ---  Identidad de Parseval (Punto b) ---
# Potencia en el tiempo
pot_tiempo0 = np.sum(s0**2) / nn  
pot_tiempo1 = np.sum(s1**2) / nn
pot_tiempo2 = np.sum(s2**2) / nn
    
# FFT con padding
# Dividimos por N para que la magnitud sea independiente del largo de la señal
XX_sp0 = np.fft.fft(sp0) / nn
XX_sp1 = np.fft.fft(sp1) / nn
XX_sp2 = np.fft.fft(sp2) / nn   
# Potencia en frecuencia (Suma de los cuadrados de los bins)
# Dividimos por (1 + cant_padds) para compensar los bins adicionales
factor_padding = 1 + cant_padds

pot_frec0 = np.sum(np.abs(XX_sp0)**2) / factor_padding
pot_frec1 = np.sum(np.abs(XX_sp1)**2) / factor_padding
pot_frec2 = np.sum(np.abs(XX_sp2)**2) / factor_padding

    
# Espectro unilateral para graficar (multiplicamos por 2 para conservar energía)
psd_sp0 = np.abs(XX_sp0)**2
psd_sp0[1:nn//2] *= 2
frec_s = np.arange(0, fs/2, df)


psd_sp1 = np.abs(XX_sp1)**2
psd_sp1[1:nn//2] *= 2 


psd_sp2 = np.abs(XX_sp2)**2
psd_sp2[1:nn//2] *= 2 

# %% Generación de gráficos con padding 

# Parámetros de frecuencia para el padding
# N_total ya lo definiste como len(sp0)
frec_sf = np.linspace(0, fs/2, N_total//2)

# Lista de datos  (usando las señales con padding y las originales)
datos = [
    (s0, sp0, XX_s0, f0, pot_tiempo0, pot_frec0, "Bin exacto"),
    (s1, sp1, XX_s1, f1, pot_tiempo1, pot_frec1, "Desintonía leve"),
    (s2, sp2, XX_s2, f2, pot_tiempo2, pot_frec2, "Máxima fuga")
]

for s_orig, s_pad, XX_orig, freq, p_t, p_f, nombre in datos:
    plt.figure(figsize=(10, 4))
    
    # 1. Respuesta Continua (Línea Azul con Zero Padding)
    # Calculamos la FFT de la señal con padding ya normalizada
    XX_pad = np.fft.fft(s_pad) / (nn/2) 
    psd_pad = np.abs(XX_pad[:N_total//2])
    plt.plot(frec_sf, 20 * np.log10(psd_pad + 1e-12), color='royalblue', label='Respuesta Continua (Zero Padding)')
    
    # 2. Bins originales de la DFT (Puntos Rojos)
    # Usamos la FFT de la señal original (sin padding)
    psd_puntos = np.abs(XX_orig[:nn//2] / (1/2))
    plt.plot(frec_s, 20 * np.log10(psd_puntos + 1e-12), 'ro', markersize=5, label='Bins DFT Original')
    
    # Títulos y formato
    plt.suptitle("Densidades Espectrales de Potencia (PDS's)", fontsize=14)
    plt.title(f"{nombre} (k0={freq/df:.2f}) | Potencia T: {p_t:.4f} F: {p_f:.4f}", fontsize=10)
    
    plt.xlim([110, 140])
    plt.xlabel("Frecuencia [Hz]")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
















    