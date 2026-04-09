import numpy as np
import matplotlib.pyplot as plt

# --- Definición de funciones auxiliares ---
def mi_funcion_sen_ruido(vmax, dc, ff, ph, nn, fs, pr):
    ts = 1/fs
    tt = np.arange(0, nn) * ts
    desvio = np.sqrt(pr)
    r = np.random.normal(0, desvio, len(tt)) 
    
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    xx_con_ruido = xx + r
    return tt, xx_con_ruido

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs
    tt = np.arange(nn) * ts
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    return tt, xx

# --- 1. Configuración de parámetros ---
fs = 1000          
N = 1000           
B = 4              
Vf = 2.0           
kn = 1             

# --- 2. Parámetros de la senoidal ---
f0 = 2.0           
fase = 0
dc = 0
A = np.sqrt(2 * 1) 

# --- 3. Cálculo de Cuantización y Ruido ---
V_range = 2 * Vf 
q = V_range / (2**B) 
Pq = (q**2) / 12
Pn = kn * Pq 

# --- 4. Generación de señales ---
tt, s = mi_funcion_sen(A, dc, f0, fase, N, fs)
tt, sr = mi_funcion_sen_ruido(A, dc, f0, fase, N, fs, Pn)

# --- 5. Proceso de Cuantización (ADC) ---
sr_quant = np.round(sr / q) * q
sr_quant = np.clip(sr_quant, -Vf, Vf - q)

# --- 6. Visualización Temporal ---
plt.figure(figsize=(16, 8)) 
plt.plot(tt, sr_quant, label='$s_Q$ (ADC out)', color='tab:blue', linewidth=2)
plt.plot(tt, sr, label='$s_R$ (ADC in)', color='tab:green', 
         linestyle=':', linewidth=1.5, marker='o', markersize=3, 
         markerfacecolor='tab:green', markeredgecolor='black', markeredgewidth=0.3)
plt.plot(tt, s, label='$s$ (analog)', color='tab:orange', linestyle=':', linewidth=1)
plt.title(f'Señal muestreada por un ADC de {B} bits - $\pm V_R = {Vf:.1f}$ V - q = {q:.3f} V')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.ylim([-Vf * 1.05, Vf * 1.05])
plt.xlim([-0.05, 1.05])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='upper right', frameon=True)
plt.show()

# --- FFT y Densidad de Potencia  ---
# 1. Señal pura (s)
XX = np.fft.fft(s) / N
XX_mag = np.abs(XX[:N//2]) 

# 2. Señal con ruido (sr)
XXr = np.fft.fft(sr) / N
XXr_mag = np.abs(XXr[:N//2])

# 3. Señal cuantizada (sr_quant)
XXq = np.fft.fft(sr_quant) / N
XXq_mag = np.abs(XXq[:N//2])

# --- Conversión a dB (Normalizada) ---
ref = np.max(XX_mag)
S_dB = 20 * np.log10(XX_mag / ref + 1e-12) 
Sr_dB = 20 * np.log10(XXr_mag / ref + 1e-12)
Sq_dB = 20 * np.log10(XXq_mag / ref + 1e-12)

# --- Cálculo de Pisos de Ruido ---
piso_analog = np.mean(Sr_dB[10:]) 
piso_digital = np.mean(Sq_dB[10:])

# --- Gráfico Final Frecuencia ---
plt.figure(4, figsize=(12, 6))
plt.clf()

freqs = np.linspace(0, fs/2, len(S_dB))

plt.plot(freqs, Sq_dB, label='$s_Q$ - ADC out', color='tab:blue', linewidth=1.5)
plt.plot(freqs, Sr_dB, label='$s_R$ (ADC in)', color='tab:green', linestyle='dotted', alpha=0.7)
plt.plot(freqs, S_dB, label='$s$ (analog)', color='orange', linestyle='--', alpha=0.5)

plt.axhline(piso_analog, color='red', linestyle='--', label=f'piso analógico: {piso_analog:.1f} dB')
plt.axhline(piso_digital, color='cyan', linestyle='--', label=f'piso digital: {piso_digital:.1f} dB')

plt.title(f'Espectro de Potencia - ADC {B} bits')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [dB]')
plt.xlim([0, fs/2])
plt.ylim([-100, 10]) 
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.show()


# ---  Graficación: Análisis Estadístico  ---
# Usamos el error calculado: ee = sr_quant - sr
ee = sr_quant - sr

# Definimos un solo eje (ax)
plt.figure(figsize=(10, 6))

# Dibujamos el Histograma
# 'density=True' para que la altura coincida con 1/q
n, bins, patches = plt.hist(ee, bins=10 , density=True, color='#2e8b57', edgecolor='white', linewidth=0.5)

# La altura teórica de una distribución uniforme entre -q/2 y q/2 es 1/q
plt.axhline(1/q, color='orange', linestyle=':', linewidth=2, label=f'PDF Uniforme (1/q ≈ {1/q:.2f})')

# Límites laterales en -q/2 y q/2 para marcar dónde debería terminar el error
plt.axvline(-q/2, color='red', linestyle='--', alpha=0.5)
plt.axvline(q/2, color='red', linestyle='--', alpha=0.5)

plt.title("Histograma del error de cuantización ($e = s_Q - s_R$)")
plt.xlabel("Error [V]")
plt.ylabel("Densidad de probabilidad")
plt.legend()
plt.grid(True, alpha=0.2)

plt.show()