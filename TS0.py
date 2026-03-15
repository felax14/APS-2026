"""
La primer tarea consistirá en programar una función que genere señales senoidales y que permita parametrizar:

la amplitud máxima de la senoidal (volts)
su valor medio (volts)
la frecuencia (Hz)
la fase (radianes)
la cantidad de muestras digitalizada por el ADC (# muestras)
la frecuencia de muestreo del ADC.
es decir que la función que uds armen debería admitir se llamada de la siguiente manera

tt, xx = mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs)


Bonus:

💎 Ser el primero en subir un enlace a tu notebook en esta tarea
Realizar los experimentos que se comentaron en clase. Siguiendo la notación de la función definida más arriba:
ff = 500 Hz
ff = 999 Hz
ff = 1001 Hz
ff = 2001 Hz
🤯 Implementar alguna otra señal propia de un generador de señales. 

"""
import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs
    tt = np.arange(nn) * ts
    
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    
    return tt, xx

# --- Parámetros para ver 1 Hz ---
fs = 100        # Muestreamos 100 veces por segundo (suficiente para 1 Hz)
N = 100         # Tomamos 100 muestras (así tenemos exactamente 1 segundo)
vmax = 1        # 1 Volt de amplitud
dc = 0          # Sin offset
f1 = 1          # frecuencia de la senal 1 Hz
fase = 0        # 0 radianes

# Definimos la funcion de 1Hz
tt, xx = mi_funcion_sen(vmax=vmax, dc=dc, ff=f1, ph=fase, nn=N, fs=fs)

# Graficamos f1 = 1 Hz
plt.figure(figsize=(10, 4))
plt.plot(tt, xx, 'b-o', markersize=3, label=f'Senoidal {f1} Hz')
plt.title(f"Señal Senoidal de {f1} Hz (1 ciclo completo)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()
plt.show()

# Definimos la funcion de 500Hz
tt, xx = mi_funcion_sen(vmax=vmax, dc=dc, ff=500, ph=fase, nn=N, fs=fs)

# Graficamos f1 = 500 Hz
plt.figure(figsize=(10, 4))
plt.plot(tt, xx, 'b-o', markersize=3, label=f'Senoidal {500} Hz')
plt.title(f"Señal Senoidal de {500} Hz ")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()
plt.show()

# Definimos la funcion de 999Hz
tt, xx = mi_funcion_sen(vmax=vmax, dc=dc, ff=999, ph=fase, nn=N, fs=fs)

# Graficamos f1 = 999 Hz
plt.figure(figsize=(10, 4))
plt.plot(tt, xx, 'b-o', markersize=3, label=f'Senoidal {999} Hz')
plt.title(f"Señal Senoidal de {999} Hz ")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()
plt.show()

# Definimos la funcion de 1001Hz
tt, xx = mi_funcion_sen(vmax=vmax, dc=dc, ff=1001, ph=fase, nn=N, fs=fs)

# Graficamos f1 = 500 Hz
plt.figure(figsize=(10, 4))
plt.plot(tt, xx, 'b-o', markersize=3, label=f'Senoidal {1001} Hz')
plt.title(f"Señal Senoidal de {1001} Hz ")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()
plt.show()

# Definimos la funcion de 2001Hz
tt, xx = mi_funcion_sen(vmax=vmax, dc=dc, ff=2001, ph=fase, nn=N, fs=fs)

# Graficamos f1 = 2001 Hz
plt.figure(figsize=(10, 4))
plt.plot(tt, xx, 'b-o', markersize=3, label=f'Senoidal {2001} Hz')
plt.title(f"Señal Senoidal de {2001} Hz ")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()
plt.show()





