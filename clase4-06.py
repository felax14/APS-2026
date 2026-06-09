#%% Librerías
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal as sig
import scipy.io as sio

#%% Ejemplo 1: Módulo en veces y Fase desenvuelta
fs = 1000      # Frecuencia de muestreo en Hz
# wp = 70       # Banda de paso en Hz
# ws = 100      # Banda de atenación en Hz
gpass = 1     # Atenuación máxima permitida en banda de paso (dB)
gstop = 40    # Atenuación mínima requerida en banda de stop (dB)

# Para un filtro pasa banda (diseno de plantilla)
ws1 = .1
wp1 = .5
ws2 = 45
wp2 = 35

wp = [wp1, wp2]
ws = [ws1, ws2] 

ftype = 'butter'
#ftype = 'cheby1'
#ftype = 'cheby2'
#ftype = 'cauer'

# b_coeffs, a_coeffs = sig.iirdesign(wp, ws, gpass, gstop, fs=fs, analog=False, ftype= ftype, output='ba')
# taps = b_coeffs.shape[0]

ww = np.concatenate([
    np.logspace(start=-2, stop=0.1, num=500),
    np.linspace(start=1.26, stop=35, num=200),
    np.logspace(start=1.55, stop=1.65, num=300),
    np.linspace(start=46, stop=fs//2, num=50)
])

#%% DISENO DE FIRs (edicion en clase 4-06)


numtaps = 3600 # (FILTRO TIPO 2)

# gains = 10**((-1) * np.array([gstop, gstop, gpass, gpass, gstop, gstop]) / 20)
gains = np.array([0, 0, 1, 1, 0, 0])

## if numtaps % 2 == 0:
##   gains[-1] = 0.
    
b_win = sig.firwin2(
    numtaps, 
    freq = np.array([0., ws1, wp1, wp2, ws2, fs//2]),
    gain = gains, 
    window = 'boxcar',
    fs = fs)

w, h = sig.freqz(b_win, worN=ww, fs=fs)

ceros, polos, k = sig.tf2zpk(b_win, a=1)

#%% =============================================================================
# DIAGRAMA DE POLOS Y CEROS (Corrección de sos2zpk a tf2zpk para FIR)
# =============================================================================

phase = np.unwrap(np.angle(h))

fig_sys = plt.figure(figsize=(12, 5), tight_layout=True)
ax_z = fig_sys.add_subplot(1, 2, 1)

# Al ser un FIR, pasamos la transferencia como b_win y a=1
ceros, polos, k = sig.tf2zpk(b_win, 1)

circulo = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1.5)
ax_z.add_artist(circulo)

ax_z.plot(np.real(ceros), np.imag(ceros), 'bo', markersize=8, fillstyle='none', label='Ceros')
ax_z.plot(np.real(polos), np.imag(polos), 'rx', markersize=8, mew=2, label='Polos')

ax_z.axhline(0, color='black', linewidth=0.5)
ax_z.axvline(0, color='black', linewidth=0.5)
ax_z.set_title('Diagrama de Polos y Ceros (Plano z) - FIR')
ax_z.set_xlabel('Parte Real')
ax_z.set_ylabel('Parte Imaginaria')
ax_z.axis('equal')
ax_z.set_xlim([-1.2, 1.2])
ax_z.set_ylim([-1.2, 1.2])
ax_z.grid(True, alpha=0.5)
ax_z.legend()

# =============================================================================
# DIAGRAMA DE RETARDO DE GRUPO
# =============================================================================
# Panel Derecho: Retardo de Grupo
ax_gd = fig_sys.add_subplot(1, 2, 2)

# Calcular el retardo de grupo usando scipy
# sos2tf convierte la matriz sos a polinomios b, a solo para el cálculo rápido del retardo

# w_gd, gd = sig.group_delay(sig.sos2tf(sos), w=1024, fs=fs)

gd = -np.diff(phase)/np.diff(2*np.pi*ww/fs)
gd = np.append(gd[0], gd)

ax_gd.plot(ww, gd, 'm', linewidth=2)
ax_gd.set_title('Retardo de Grupo (Group Delay)')
ax_gd.set_xlabel('Frecuencia [Hz]')
ax_gd.set_ylabel('Retardo [Muestras]')
ax_gd.set_xlim(0, fs/2)
ax_gd.set_ylim(0, 1.1 * np.max(gd))
ax_gd.grid(True, alpha=0.5)

plt.show()

# =============================================================================
# BLOQUE DE PLOT CON PLANTILLA DE DISEÑO (SOMBRADA)
# =============================================================================

fig, ax1 = plt.subplots(figsize=(12, 5), tight_layout=True)
ax1.set_title(f"Frequency Response of FIR Filter ")

# 1. Dibujar la curva del filtro original (Magnitud en dB)
ax1.plot(w, 20 * np.log10(np.abs(h)), 'b', linewidth=1.8, label='Filtro diseñado')

# 2. Sombreado de la Plantilla (Zonas prohibidas / tolerancias)
# Piso y techo visual del gráfico para los rellenos
piso_grafico = -125
techo_grafico = 10

# --- Banda de parada 1 (0 a ws1) ---
ax1.fill_between([0, ws1], -gstop, techo_grafico, color='green', alpha=0.15, label='Plantilla')
ax1.plot([0, ws1], [-gstop, -gstop], 'k--', linewidth=1, alpha=0.7) # Línea de trazo límite

# --- Banda de paso (wp1 a wp2) ---
# Zona inferior (Atenuación máxima permitida)
ax1.fill_between([wp1, wp2], piso_grafico, -gpass, color='green', alpha=0.15)
ax1.plot([wp1, wp2], [-gpass, -gpass], 'k--', linewidth=1, alpha=0.7)
# Zona superior (Margen por encima de 0dB, ej: 3dB para evitar rizado excesivo)
ax1.fill_between([wp1, wp2], 3, techo_grafico, color='green', alpha=0.15)
ax1.plot([wp1, wp2], [3, 3], 'k--', linewidth=1, alpha=0.7)

# --- Banda de parada 2 (ws2 a Nyquist) ---
ax1.fill_between([ws2, fs/2], -gstop, techo_grafico, color='green', alpha=0.15)
ax1.plot([ws2, fs/2], [-gstop, -gstop], 'k--', linewidth=1, alpha=0.7)

# Líneas verticales que delimitan los saltos de las bandas
ax1.axvline(ws1, color='k', linestyle=':', alpha=0.5)
ax1.axvline(wp1, color='k', linestyle=':', alpha=0.5)
ax1.axvline(wp2, color='k', linestyle=':', alpha=0.5)
ax1.axvline(ws2, color='k', linestyle=':', alpha=0.5)

# 3. Configuración estricta de límites y estética
ax1.set_ylabel('Amplitude in dB', color='b')
ax1.set_xlabel('Frequency [Hz]')
ax1.set_xlim(0, fs/2)
ax1.set_ylim([piso_grafico, techo_grafico]) 

ax1.grid(True, which='both', linestyle='-', alpha=0.4)
ax1.legend(loc='lower right')

plt.show()


