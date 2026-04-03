"""Sintetizar:

1.Señal sinusoidal de 2KHz.
2.Misma señal amplificada 3 dB y desfasada en π/2.
3.Misma señal modulada en amplitud por otra señal sinusoidal de frecuencia de 1000 KHz.
4.Misma señal con efecto de saturación al 75% de su amplitud. Ayuda: ver numpy.clip().
5.Una señal cuadrada de 4KHz.
6.Un pulso rectangular de 10ms.
En cada caso indique tiempo entre muestras, número de muestras y potencia o energía según corresponda."""



import numpy as np
import matplotlib.pyplot as plt


#%%
def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs
    tt = np.arange(nn) * ts
    
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    
    return tt, xx
#%%
def mi_funcion_cuadrada(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs
    tt = np.arange(nn) * ts
    
    # Generamos la base senoidal con la fase y frecuencia
    # np.sign devuelve 1 para valores positivos y -1 para negativos
    base_seno = np.sin(2 * np.pi * ff * tt + ph)
    xx = vmax * np.sign(base_seno) + dc
    
    return tt, xx
#%%
def mi_pulso_rectangular(vmax, dc, ancho_pulse, ph, nn, fs):
    ts = 1/fs
    tt = np.arange(nn) * ts
    
    # Creamos un array de ceros (o del valor DC)
    xx = np.full(nn, dc, dtype=float)
    
    # Definimos el inicio y fin del pulso
    # ph (fase) aquí funcionaría como el retraso inicial (delay)
    t_inicio = ph
    t_fin = ph + ancho_pulse
    
    # Activamos el pulso donde el tiempo esté en el rango
    # Usamos una máscara booleana para mayor claridad
    indices_pulso = (tt >= t_inicio) & (tt <= t_fin)
    xx[indices_pulso] = vmax + dc
    
    return tt, xx


#%%
# ---  Generación de señales ---

#  1.Señal sinusoidal de 2KHz.
# --- Parámetros--- 
fs = 40000      
N =  100      
vmax = 1        
dc = 0          
f1 = 2000          
fase = 0       

# Definimos la funcion seno de 2Hz
tt, xx = mi_funcion_sen(vmax=vmax, dc=dc, ff=f1, ph=fase, nn=N, fs=fs)
# Calculo de tiempo entre muestras, número de muestras y potencia o energía según corresponda.
print(f"Punto 1: Ts = {1/fs}s, N = {len(xx)}, Potencia = {np.mean(xx**2):.4f}W")

# Graficamos f1 = 2 kHz
plt.figure(figsize=(10, 4))
plt.plot(tt, xx, markersize=3, label='Senoidal 2 kHz')
plt.title("Señal Senoidal de 2 kHz ")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()
plt.show()


#%%
# 2.Misma señal amplificada en 3dB y desfasada en π/2.
# Definimos la funcion seno amplificada (10^(3/20) aprox 1.4125) y desfasada.
tt1, xx2 = mi_funcion_sen(vmax=vmax * (10**(3/20)), dc=dc, ff=f1, ph=np.pi/2, nn=N, fs=fs)

# Calculo de tiempo entre muestras, número de muestras y potencia o energía según corresponda.
print(f"Punto 2: Ts = {1/fs}s, N = {len(xx2)}, Potencia = {np.mean(xx2**2):.4f}W")

# Graficamos funcion punto 2
plt.figure(figsize=(10, 4))
plt.plot(tt1, xx2, markersize=3, label='Senoidal 2 kHz (+3dB)')
plt.title("Senoidal 2 kHz Desfasada y Amplificada 3dB")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()
plt.show()


#%%
# 3.Misma señal modulada en amplitud por otra señal sinusoidal de la mitad de la frecuencia.
#  Generamos la moduladora (información) - f1/2 = 1000 Hz


Am = 2 # Amplitud de la información (el valor pico)
ka = 1.0 # Índice de modulación al 100% (m=1)


# Generamos la MODULADORA (información) 'A(n)' - f1/2 = 1000 Hz.
tt_am, A_n = mi_funcion_sen(vmax=Am, dc=Am, ff=f1/2, ph=0, nn=N, fs=fs)

# Generamos la portadora - f1 = 2000 Hz
_, portadora = mi_funcion_sen(vmax=1, dc=0, ff=f1, ph=0, nn=N, fs=fs)

# Fórmula de Modulación AM: Multiplicación de envolvente * portadora.
modulacion_am = A_n * portadora

# Calculo de tiempo entre muestras, número de muestras y potencia o energía según corresponda.
print(f"Punto 3: Ts = {1/fs}s, N = {len(modulacion_am)}, Potencia = {np.mean(modulacion_am**2):.4f}W")

# Graficamos 
plt.figure(figsize=(10, 5))
plt.plot(tt, modulacion_am, 'b', linewidth=0.8, label='Señal AM')
plt.plot(tt, A_n, 'r--', alpha=0.6, label='Envolvente A(n)')
plt.plot(tt, -A_n, 'r--', alpha=0.6, label='Envolvente negativa')
plt.title('Modulación en Amplitud - Recreación del Cuaderno')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.axhline(0, color='black', linewidth=1)
plt.grid(True)
plt.legend()
plt.show()


#%%
# 4.Señal anterior recortada al 75% de su amplitud.
# Definimos el umbral de recorte (75% de vmax)
umbral = vmax * 0.75  # En este caso, 0.75V

# Aplicamos el recorte a la señal original xx (la de 2000 Hz)
# np.clip(array, min, max) limita los valores al rango especificado
xx_recortada = np.clip(xx, -umbral, umbral)

# Calculo de tiempo entre muestras, número de muestras y potencia o energía según corresponda.
print(f"Punto 4: Ts = {1/fs}s, N = {len(xx_recortada)}, Potencia = {np.mean(xx_recortada**2):.4f}W")

# Graficamos la funcion recortada
plt.figure(figsize=(10, 4))
plt.plot(tt, xx_recortada, markersize=3, label='Senoidal senoidal')
plt.title("Senoidal Recortada al 75% de amplitud")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()
plt.show()


#%%
# 5.Una señal cuadrada de 4KHz.
# Definimos la funcion cuadrada de 4 kHz
tt_cuad, xx3 = mi_funcion_cuadrada(vmax=vmax, dc=dc, ff=4000, ph=fase, nn=N, fs=fs)

# Calculo de tiempo entre muestras, número de muestras y potencia o energía según corresponda.
print(f"Punto 5: Ts = {1/fs}s, N = {len(xx3)}, Potencia = {np.mean(xx3**2):.4f}W")

# Graficamos la funcion cuadrada 
plt.figure(figsize=(10, 4))
plt.plot(tt_cuad, xx3, 'r', linewidth=2, label='Cuadrada 4 kHz')
plt.title("5. Señal Cuadrada de 4 kHz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.xlim(0, 0.002) 
plt.grid(True)
plt.legend()
plt.show()


#%%
# 6.Un pulso rectangular de 10ms.
# Definimos el pulso de 10 ms
tt_pulso, xx4 = mi_pulso_rectangular(vmax=vmax, dc=dc, ancho_pulse=0.01, ph=0.005, nn=1000, fs=fs)

# Calculo de tiempo entre muestras, número de muestras y potencia o energía según corresponda.
print(f"Punto 6: Ts = {1/fs}s, N = {len(xx4)}, Energía = {np.sum(xx4**2)*(1/fs):.4f}J")

# Graficamos el pulso
plt.figure(figsize=(10, 4))
plt.plot(tt_pulso, xx4, 'g', linewidth=2, label='Pulso 10ms')
plt.title("6. Pulso Rectangular de 10ms")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()
plt.show()
#%%



