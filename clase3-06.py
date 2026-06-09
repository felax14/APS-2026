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

sos = sig.iirdesign(wp, ws, gpass, gstop, fs=fs, analog=False, ftype=ftype, output='sos')
# Nota: Quitamos la variable 'taps' porque ya no tenemos un vector directo b_coeffs.

# Respuesta en frecuencia (w saldrá en Hz por usar fs=fs)
# w, h = sig.freqz(b_coeffs, a_coeffs, worN=1024, fs=fs)

ww = np.concatenate([
    np.logspace(start=-2, stop=0.1, num=500),
    np.linspace(start=1.26, stop=35, num=200),
    np.logspace(start=1.55, stop=1.65, num=300),
    np.linspace(start=46, stop=fs//2, num=50)
])

# w, h = sig.sosfreqz(sos, worN=1024, fs=fs)

w, h = sig.sosfreqz(sos, worN=ww, fs=fs)

fig, ax1 = plt.subplots(tight_layout=True)
ax1.set_title(f"Respuesta en Frecuencia del Filtro IIR ")

# Gráfico del Módulo (En veces de forma lineal)
ax1.plot(w, abs(h), 'C0', label='Módulo')
ax1.set_ylabel("Módulo [Veces]", color='C0')
ax1.set_xlabel("Frecuencia [Hz]")
# ax1.set_xlim(0, fs/2) # El límite físico es Nyquist (fs/2)
ax1.tick_params(axis='y', labelcolor='C0')
ax1.grid(True)

# Gráfico de la Fase (En radianes)
ax2 = ax1.twinx()
phase = np.unwrap(np.angle(h)) 
ax2.plot(w, phase, 'C1', label='Fase')
ax2.set_ylabel('Fase [rad]', color='C1')
ax2.tick_params(axis='y', labelcolor='C1')

plt.show()

#%% Ejemplo 2: Respuesta con Módulo en Decibeles [dB] e índices alineados
# Usamos las mismas variables calculadas arriba



fig, ax1 = plt.subplots(tight_layout=True)
ax1.set_title('Respuesta en Frecuencia Digital (Escala dB)')

# Magnitud en dB
ax1.plot(w, 20 * np.log10(abs(h)), 'b')
ax1.set_ylabel('Amplitud [dB]', color='b')
ax1.set_xlabel('Frecuencia [Hz]')
# ax1.set_xlim(0, fs/2)
# ax1.set_ylim([-60, 5]) # Ajustado el límite inferior para apreciar la caída
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# Fase desenvuelta
ax2 = ax1.twinx()
phase = np.unwrap(np.angle(h))
ax2.plot(w, phase, 'g')
ax2.set_ylabel('Fase [rad]', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Alinear la cantidad de grillas (ticks) de ambos ejes para que se vea prolijo
nticks = 6
ax1.yaxis.set_major_locator(ticker.LinearLocator(nticks))
ax2.yaxis.set_major_locator(ticker.LinearLocator(nticks))
ax2.grid(True, linestyle='--', alpha=0.5)

plt.show()

# =============================================================================
# DIAGRAMA DE POLOS Y CEROS
# =============================================================================
fig_sys = plt.figure(figsize=(12, 5), tight_layout=True)

# Panel Izquierdo: Plano Z
ax_z = fig_sys.add_subplot(1, 2, 1)

# Calcular polos y ceros
ceros, polos, k = sig.sos2zpk(sos)

# Dibujar el círculo unitario
circulo = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1.5)
ax_z.add_artist(circulo)

# Graficar ceros (o) y polos (x)
ax_z.plot(np.real(ceros), np.imag(ceros), 'bo', markersize=8, fillstyle='none', label='Ceros')
ax_z.plot(np.real(polos), np.imag(polos), 'rx', markersize=8, mew=2, label='Polos')

ax_z.axhline(0, color='black', linewidth=0.5)
ax_z.axvline(0, color='black', linewidth=0.5)
ax_z.set_title('Diagrama de Polos y Ceros (Plano z)')
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

# Calculamos el retardo como la derivada de la fase 
gd = - np.diff(phase) / np.diff(ww)
gd = np.append(gd[0], gd)

# Ahora ww y gd tienen exactamente el mismo tamaño (num_puntos)
ax_gd.plot(ww, gd, 'm', linewidth=2)
ax_gd.set_title('Retardo de Grupo (Group Delay)')
ax_gd.set_xlabel('Frecuencia [Hz]')
ax_gd.set_ylabel('Retardo [Muestras]')
# ax_gd.set_xlim(0, fs/2)
ax_gd.grid(True, alpha=0.5)

plt.show()

# =============================================================================
# BLOQUE DE PLOT CON PLANTILLA DE DISEÑO (SOMBRADA)
# =============================================================================

fig, ax1 = plt.subplots(figsize=(12, 5), tight_layout=True)
ax1.set_title(f"Frequency Response of IIR Filter (SOS Structure)")

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
#%%
# =============================================================================
# IMPLEMENTACION Y COMPARACIÓN DE FILTROS EN SEÑAL ECG
# =============================================================================

fs_ecg = 1000 # Hz

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)

# Definimos los tipos de filtros IIR solicitados
tipos_filtros = ['butter', 'cauer', 'cheby2', 'cheby1'] 

# Diccionario para almacenar las señales procesadas por cada filtro
ecg_filtrados = {}

# Ejecutamos el diseño y filtrado de forma secuencial (utilizando la plantilla armada arriba)
for ftype in tipos_filtros:
    # Divido por dos las atenuaciones debido a que sosfiltfilt procesa la señal dos veces
    sos_ff = sig.iirdesign(wp, ws, gpass/2, gstop/2, fs=fs, analog=False, ftype=ftype, output='sos')
    
    # Procesamos la señal con filtrado de fase cero
    ecg_filtrados[ftype] = sig.sosfiltfilt(sos_ff, ecg_one_lead)

# =============================================================================
# 1. GRÁFICOS DE LAS REGIONES DE INTERÉS SIN RUIDO
# =============================================================================
print("\n--- Desplegando Regiones de Interés SIN RUIDO ---")

regs_interes_sin_ruido = (
        np.array([5, 5.2]) * 60 * fs_ecg, # minutos a muestras
        np.array([12, 12.4]) * 60 * fs_ecg, # minutos a muestras
        np.array([15, 15.2]) * 60 * fs_ecg, # minutos a muestras
        )

for idx, ii in enumerate(regs_interes_sin_ruido):
   
    # intervalo limitado de 0 a cant_muestras (N)
    inicio = int(np.max([0, ii[0]]))
    fin = int(np.min([N, ii[1]]))
    zoom_region = np.arange(inicio, fin, dtype='uint')
   
    # Creamos una ventana nueva limpia por cada iteración
    plt.figure()
    
    # 1. Graficar señal original (ECG)
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    
    # 2. Graficar llamando a las señales 
    plt.plot(zoom_region, ecg_filtrados['butter'][zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_filtrados['cauer'][zoom_region], label='Cauer')
    plt.plot(zoom_region, ecg_filtrados['cheby2'][zoom_region], label='Cheby2')
    plt.plot(zoom_region, ecg_filtrados['cheby1'][zoom_region], label='Cheby1')
   
    # Etiqueta agregada al título: "SIN RUIDO - Región X"
    plt.title(f'ECG SIN RUIDO (Región {idx+1}) - Samples {inicio} to {fin}')
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.xlim(inicio, fin)
   
    # Estética idéntica a tu captura (Sin grillas y leyenda abajo al centro)
    axes_hdl = plt.gca()
    axes_hdl.legend(loc='lower center', framealpha=1.0)
    axes_hdl.set_yticks(())
    axes_hdl.grid(False) 
            
    plt.show()
    
# =============================================================================
# 2. GRÁFICOS DE LAS REGIONES DE INTERÉS CON RUIDO
# =============================================================================
print("\n--- Desplegando Regiones de Interés CON RUIDO ---")

regs_interes_con_ruido = (
        [4000, 5500], # muestras
        [10000, 11000], # muestras (pasados a int planos)
        )

for idx, ii in enumerate(regs_interes_con_ruido):
   
    # intervalo limitado de 0 a cant_muestras (N)
    inicio = int(np.max([0, ii[0]]))
    fin = int(np.min([N, ii[1]]))
    zoom_region = np.arange(inicio, fin, dtype='uint')
   
    # Creamos una ventana nueva limpia por cada iteración
    plt.figure()
    
    # 1. Graficar señal original (ECG)
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    
    # 2. Graficar llamando a las señales 
    plt.plot(zoom_region, ecg_filtrados['butter'][zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_filtrados['cauer'][zoom_region], label='Cauer')
    plt.plot(zoom_region, ecg_filtrados['cheby2'][zoom_region], label='Cheby2')
    plt.plot(zoom_region, ecg_filtrados['cheby1'][zoom_region], label='Cheby1')
   
    # Etiqueta agregada al título: "CON RUIDO - Región X"
    plt.title(f'ECG CON RUIDO (Región {idx+1}) - Samples {inicio} to {fin}')
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.xlim(inicio, fin)
   
    # Estética idéntica a tu captura (Sin grillas y leyenda abajo al centro)
    axes_hdl = plt.gca()
    axes_hdl.legend(loc='lower center', framealpha=1.0)
    axes_hdl.set_yticks(())
    axes_hdl.grid(False) 
            
    plt.show()