import numpy as np
import matplotlib.pyplot as plt

# --- Definición de funciones auxiliares ---
def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs
    tt = np.arange(nn) * ts
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    return tt, xx

# Parámetros de la senoidal
vmax = np.sqrt(2)  # energía unitaria
dc = 0
nn = 1000          # número de muestras
fs = 1000          # frecuencia de muestreo según la consigna.
ff = 1
#%%
# Señal senoidal limpia.
tt, s = mi_funcion_sen(vmax, dc, ff, ph=0, nn=nn, fs=fs)

# %% Parámetros del ADC para calcular la potencia de cuantización
B = 4             # bits
Vf = 2             # Voltios, rango analógico ±VF
q = (2*Vf)/(2**B)  # paso de cuantización
Pq = q**2 / 12     # potencia de cuantización
kn = 1        # factor de escala del ruido
Pn = kn * Pq       # potencia del ruido

# %% Ruido aditivo Gaussiano
sr = np.random.normal(0, np.sqrt(Pn), nn)
srq = s + sr    # Defino la señal con ruido.

#%%
# ---  Proceso de Cuantización (ADC) ---
# sr_quant es la señal srq (señal ruidosa cuantizada)
sr_quant = np.round(srq / q) * q
sr_quant = np.clip(sr_quant, -Vf, Vf - q)

ee = sr_quant - srq  # error sobre señal con ruido

# ---  Visualización Temporal ---
plt.figure(figsize=(16, 8)) 
plt.plot(tt, sr_quant, label='$s_Q$ (ADC out)', color='tab:blue', linewidth=2)
plt.plot(tt, srq, label='$s_R$ (ADC in)', color='tab:green', 
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
# %%
# --- FFT y Densidad de Potencia  ---
# 1. Señal pura (s)
XX = np.fft.fft(s) / nn
XX_mag = np.abs(XX[:nn//2]) 

# 2. Señal ruido gaussiano (sr)
XXr = np.fft.fft(sr) / nn
XXr_mag = np.abs(XXr[:nn//2])

# 3. Señal cuantizada (sr_quant)
XXq = np.fft.fft(sr_quant) / nn
XXq_mag = np.abs(XXq[:nn//2])

# 4. Errr de cuantizacion 
XXe = np.fft.fft(ee) / nn
XXe_mag = np.abs(XXe[:nn//2])

# 4. Senal pura + ruido  
XXsrq = np.fft.fft(srq) / nn
XXsrq_mag = np.abs(XXsrq[:nn//2])

# --- Conversión a dB (Normalizada) ---
# La referencia sigue siendo el pico de la señal pura
ref = np.max(XX_mag) 
S_dB  = 20 * np.log10(XX_mag / ref + 1e-12) 
Sr_dB = 20 * np.log10(XXr_mag / ref + 1e-12)
Sq_dB = 20 * np.log10(XXq_mag / ref + 1e-12)
Se_dB = 20 * np.log10(XXe_mag / ref + 1e-12)
Ssrq_dB = 20 * np.log10(XXsrq_mag / ref + 1e-12)

# --- Cálculo de Pisos de Ruido (PSD Media) ---
piso_analog  = np.mean(Sr_dB[10:]) 
piso_digital = np.mean(Sq_dB[10:])
MediaError    = np.mean(Se_dB[10:])
#%% Calculos auxiliares de verificacion
print("-" * 45)
print("Calculos auxiliares de verificacion" )
print(f"{'Piso de Ruido Analógico:':<30} {piso_analog:>8.2f} dB")
print(f"{'Piso de Ruido Digital:':<30} {piso_digital:>8.2f} dB")
print(f"{'Piso del Error de Cuantización:':<30} {MediaError:>8.2f} dB")
print("-" * 45)

# --- Cálculo de Estadísticos (Media y Varianza) ---

# 1. Señal Pura (s)
mean_s = np.mean(s)
var_s = np.var(s)  # Potencia de la señal (debería ser 1.0)

# 2. Ruido Analógico (sr)
mean_sr = np.mean(sr)
var_sr = np.var(sr) # Potencia del ruido gaussiano inyectado

# 3. Señal con Ruido (srq)
mean_srq = np.mean(srq)
var_srq = np.var(srq)

# 4. Error de Cuantización (ee = sr_quant - srq)
mean_ee = np.mean(ee)
var_ee = np.var(ee) # Potencia del error (debería ser cercana a Pq)

# --- Verificación Teórica ---
# Relación de potencias en dB para comparar con los pisos del gráfico
# Se usa (N/2) como factor de ganancia de procesamiento de la FFT
piso_esperado_ee = 10 * np.log10(var_ee / (nn/2) + 1e-12)

# --- Print de Verificación Detallada ---
print("-" * 65)
print(f"{'Señal':<25} | {'Media':<12} | {'Varianza (Pot)':<15}")
print("-" * 65)
print(f"{'Seno Puro (s)':<25} | {mean_s:>12.2e} | {var_s:>15.6f}")
print(f"{'Ruido Analógico (sr)':<25} | {mean_sr:>12.2e} | {var_sr:>15.6f}")
print(f"{'Error Cuantización (ee)':<25} | {mean_ee:>12.2e} | {var_ee:>15.6f}")
print("-" * 65)
print(f"Potencia Teórica de Cuantización (Pq): {Pq:.6f}")
print(f"Diferencia Medida vs Teórica:         {abs(var_ee - Pq):.2e}")
print("-" * 65)
print(f"Piso Espectral Calculado (Se_dB):      {MediaError:.2f} dB")
print(f"Piso Teórico Estimado (10*log10(P/N)): {piso_esperado_ee:.2f} dB")
print("-" * 65)

#%%
# --- Gráfico Final Frecuencia ---
plt.figure(4, figsize=(12, 6))
plt.clf()
freqs = np.linspace(0, fs/2, len(S_dB))
# Trazas de las señales
plt.plot(freqs, Sq_dB, label=' $s_Q$ (ADC out)', color='tab:blue', linewidth=1.5, zorder=3)
plt.plot(freqs, Se_dB, label=' $s_e$ (Error)', color='tab:red', linewidth=1.2, alpha=0.8)
plt.plot(freqs, Sr_dB, label=' $s_R$ (ADC in)', color='tab:green', linestyle=':', alpha=0.6)
plt.plot(freqs, S_dB,  label=' $s$ (Pure)', color='orange', linestyle='--', alpha=0.5)
# Líneas de referencia (Pisos de ruido)
# Piso analógico en verde (asociado a sr)
plt.axhline(piso_analog, color='tab:green', linestyle='--', linewidth=1, 
            label=f'Piso Analógico: {piso_analog:.1f} dB')
# Piso digital en cian (asociado a sq)
plt.axhline(piso_digital, color='tab:cyan', linestyle='-', linewidth=1.5, 
            label=f'Piso Digital Total: {piso_digital:.1f} dB')
# Media del error en rojo (asociado a se)
plt.axhline(MediaError, color='tab:red', linestyle=':', linewidth=1, 
            label=f'Piso de Cuantización: {MediaError:.1f} dB')
plt.title(f'Densidad Espectral de Potencia - ADC {B} bits')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia   [dB]') 
plt.xlim([0, fs/2])
plt.ylim([-100, 10]) 
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.show()
#%%
# ---  Graficación: Análisis Estadístico  ---
ee = sr_quant - srq

plt.figure(figsize=(10, 6))

# Histogram con bins alineados a q
n, bins, patches = plt.hist(ee, bins=10, density=True, color='#2e8b57', 
                            edgecolor='white', linewidth=0.5, label='Error medido')

# Altura teórica
plt.axhline(1/q, color='orange', linestyle=':', linewidth=2, 
            label=f'PDF Uniforme (1/q ≈ {1/q:.2f})')

# Límites de cuantización
plt.axvline(-q/2, color='red', linestyle='--', alpha=0.5, label='$\pm q/2$')
plt.axvline(q/2, color='red', linestyle='--', alpha=0.5)

plt.title("Histograma del error de cuantización ($e = s_Q - s_R$)")
plt.xlabel("Error [V]")
plt.ylabel("Densidad de probabilidad")

# Ajuste de rangos para ver mejor los límites
plt.xlim([-q, q]) 
plt.legend(loc='upper right')
plt.grid(True, alpha=0.2)

plt.show()