import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

eps = 1e-12
def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1 / fs
    tt = np.arange(0, nn) * ts
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    return tt, xx

#%% PARAMETROS 
N    = 1000
a0   = np.sqrt(2)
fs   = N
df   = fs/N
rea  = 200
SNR1  = 10
SNR2  = 3
Pr1   = 10**(-SNR1/10)
Pr2   = 10**(-SNR2/10)
ff = np.arange (N) * df
ff_zp = fs * np.arange (10*N) / (10*N)

#%% GENERO UN VECTOR DE NUMEROS ALEATORIOS y RUIDO CON SNR = 10 Y SNR =3 

R1  = np.random.normal(0, np.sqrt(Pr1), N)
R2  = np.random.normal(0, np.sqrt(Pr2), N)
fr = np.random.uniform(-2, 2, rea)

#%% ARMO LA FUNCION SENO + RUIDO 
#Con cuidado pasar omega a frec.... pi/2=n/4 2pi/n=1 y deltaf=fs/n
#w1= n/4 + fr*((2*np.pi)/n)
w1 = (N/4 + fr)*df

arr_w1 = w1.reshape(rea,1)
nn = np.arange(0,N)
arr_n = nn.reshape(1,N)

f = arr_w1*arr_n*1/fs # Matriz de frecuencias 

s = a0*np.sin(2*np.pi*f) # Creo una matriz de funciones seno 

# Funcion seno + Ruido 
x1 = s + R1
x2 = s + R2

#%% Generacion de FFTS 

# Ventaneo 
w_rect = np.ones((1, N)) # Cambiado de (N,1) a (1,N)
w_flat = sig.windows.flattop(N, sym=False).reshape(1,-1)
w_bh   = sig.windows.blackmanharris(N, sym=False).reshape(1,-1)
w_hann = sig.windows.hann(N, sym=False).reshape(1,-1)

# Ventaneo y FFT de la señal x1 (SNR = 10dB)
x1_rect = x1 * w_rect
x1_flat = x1 * w_flat
x1_bh   = x1 * w_bh
x1_hann = x1 * w_hann

X1_rect = (1/N) * np.fft.fft(x1_rect, axis=1)
X1_flat = (1/N) * np.fft.fft(x1_flat, axis=1)
X1_bh   = (1/N) * np.fft.fft(x1_bh,   axis=1)
X1_hann = (1/N) * np.fft.fft(x1_hann, axis=1)

# Zero Padding (n=10*N)
X1_rect_zp = (1/N) * np.fft.fft(x1_rect, n=10*N, axis=1)
X1_flat_zp = (1/N) * np.fft.fft(x1_flat, n=10*N, axis=1)
X1_bh_zp   = (1/N) * np.fft.fft(x1_bh,   n=10*N, axis=1)
X1_hann_zp = (1/N) * np.fft.fft(x1_hann, n=10*N, axis=1)

# Ventaneo y FFT de la señal x2 (SNR = 3dB)
x2_rect = x2 * w_rect
x2_flat = x2 * w_flat
x2_bh   = x2 * w_bh
x2_hann = x2 * w_hann

X2_rect = (1/N) * np.fft.fft(x2_rect, axis=1)
X2_flat = (1/N) * np.fft.fft(x2_flat, axis=1)
X2_bh   = (1/N) * np.fft.fft(x2_bh,   axis=1)
X2_hann = (1/N) * np.fft.fft(x2_hann, axis=1)

# Zero Padding (n=10*N)
X2_rect_zp = (1/N) * np.fft.fft(x2_rect, n=10*N, axis=1)
X2_flat_zp = (1/N) * np.fft.fft(x2_flat, n=10*N, axis=1)
X2_bh_zp   = (1/N) * np.fft.fft(x2_bh,   n=10*N, axis=1)
X2_hann_zp = (1/N) * np.fft.fft(x2_hann, n=10*N, axis=1)


#%% Plot de Ventanas para SNR = 10 dB (X1)

ff_unilateral = ff[:N//2]
plt.figure(1, figsize=(10, 10)) # Figura 1

ventanas = [X1_rect, X1_flat, X1_bh, X1_hann]
titulos = ['Rectangular', 'Flattop', 'Blackman-Harris', 'Hann']

for i, x_fft in enumerate(ventanas):
    plt.subplot(4, 1, i+1)
    # Graficamos las 200 realizaciones (transpuestas)
    plt.plot(ff_unilateral, 10*np.log10(np.abs(x_fft[:, :N//2]).T**2 + eps), linewidth=0.4, alpha=0.3)
    plt.title(f'PSD - Ventana {titulos[i]} (SNR = 10 dB)')
    plt.ylabel('[dB]')
    plt.xlim(0, fs/2)
    plt.grid(True)

plt.xlabel('Frecuencia [Hz]')
plt.tight_layout()
plt.show()

#%% Plot de Ventanas para SNR = 3 dB (X2)

plt.figure(2, figsize=(10, 10)) # Figura 2 para no sobreescribir

ventanas_2 = [X2_rect, X2_flat, X2_bh, X2_hann]

for i, x_fft in enumerate(ventanas_2):
    plt.subplot(4, 1, i+1)
    plt.plot(ff_unilateral, 10*np.log10(np.abs(x_fft[:, :N//2]).T**2 + eps), linewidth=0.4, alpha=0.3)
    plt.title(f'PSD - Ventana {titulos[i]} (SNR = 3 dB)') # Título corregido a 3 dB
    plt.ylabel('[dB]')
    plt.xlim(0, fs/2)
    plt.grid(True)

plt.xlabel('Frecuencia [Hz]')
plt.tight_layout()
plt.show()

#%% Calculo de estimadores en amplitud y frecuenicia

#Estimadores de amplitud de la señal x1 (SNR = 10dB)


ax1_rect = 2*np.max(np.abs(X1_rect), axis=0) / np.mean(w_rect)
ax1_flat = 2*np.max(np.abs(X1_flat), axis=0) / np.mean(w_flat)
ax1_bh   = 2*np.max(np.abs(X1_bh),   axis=0) / np.mean(w_bh)
ax1_hann = 2*np.max(np.abs(X1_hann), axis=0) / np.mean(w_hann)

sesgo_ax1_rect = np.mean (ax1_rect) - a0 
sesgo_ax1_flat = np.mean (ax1_flat) - a0
sesgo_ax1_bh   = np.mean (ax1_bh)   - a0
sesgo_ax1_hann = np.mean (ax1_hann) - a0

var_ax1_rect = np.var (ax1_rect) 
var_ax1_flat = np.var (ax1_flat) 
var_ax1_bh   = np.var (ax1_bh)
var_ax1_hann = np.var (ax1_hann)


#Estimadores de amplitud de la señal x2 (SNR = 3dB) 


ax2_rect = 2*np.max(np.abs(X2_rect), axis=0) / np.mean(w_rect)
ax2_flat = 2*np.max(np.abs(X2_flat), axis=0) / np.mean(w_flat)
ax2_bh   = 2*np.max(np.abs(X2_bh),   axis=0) / np.mean(w_bh)
ax2_hann = 2*np.max(np.abs(X2_hann), axis=0) / np.mean(w_hann)

sesgo_ax2_rect = np.mean (ax2_rect) - a0
sesgo_ax2_flat = np.mean (ax2_flat) - a0
sesgo_ax2_bh   = np.mean (ax2_bh)   - a0
sesgo_ax2_hann = np.mean (ax2_hann) - a0

var_ax2_rect = np.var (ax2_rect)
var_ax2_flat = np.var (ax2_flat)
var_ax2_bh   = np.var (ax2_bh)
var_ax2_hann = np.var (ax2_hann)


#Estimadores de frecuencia de la señal x1 (SNR = 10dB) 


fx1_rect = np.argmax(np.abs(X1_rect[:, :N//2]), axis=1)
fx1_flat = np.argmax(np.abs(X1_flat[:, :N//2]), axis=1)
fx1_bh   = np.argmax(np.abs(X1_bh[:, :N//2]),   axis=1)
fx1_hann = np.argmax(np.abs(X1_hann[:, :N//2]), axis=1)

frec_real_bins = N/4 + fr

sesgo_fx1_rect = np.mean(fx1_rect - frec_real_bins)
sesgo_fx1_flat = np.mean(fx1_flat - frec_real_bins)
sesgo_fx1_bh   = np.mean(fx1_bh   - frec_real_bins)
sesgo_fx1_hann = np.mean(fx1_hann - frec_real_bins)

var_fx1_rect = np.var (fx1_rect)
var_fx1_flat = np.var (fx1_flat)
var_fx1_bh   = np.var (fx1_bh)
var_fx1_hann = np.var (fx1_hann)


#Estimadores de frecuencia de la señal x2 (SNR = 3dB) 


fx2_rect = np.argmax(np.abs(X2_rect[:, :N//2]), axis=1)
fx2_flat = np.argmax(np.abs(X2_flat[:, :N//2]), axis=1)
fx2_bh   = np.argmax(np.abs(X2_bh[:, :N//2]),   axis=1)
fx2_hann = np.argmax(np.abs(X2_hann[:, :N//2]), axis=1)

sesgo_fx2_rect = np.mean(fx2_rect - frec_real_bins)
sesgo_fx2_flat = np.mean(fx2_flat - frec_real_bins)
sesgo_fx2_bh   = np.mean(fx2_bh   - frec_real_bins)
sesgo_fx2_hann = np.mean(fx2_hann - frec_real_bins)

var_fx2_rect = np.var(fx2_rect - frec_real_bins)
var_fx2_flat = np.var(fx2_flat - frec_real_bins)
var_fx2_bh   = np.var(fx2_bh   - frec_real_bins)
var_fx2_hann = np.var(fx2_hann - frec_real_bins)


#Tablas

print("\n")
print("========== Estimaciones para Señal x1 ==========")
print("{:<18} | {:>10} | {:>12}".format("Ventana", "Sesgo", "Varianza"))
print("-"*45)
print("{:<18} | {:>10.6f} | {:>12.8f}".format("Rectangular", sesgo_ax1_rect, var_ax1_rect))
print("{:<18} | {:>10.6f} | {:>12.8f}".format("Flattop", sesgo_ax1_flat, var_ax1_flat))
print("{:<18} | {:>10.6f} | {:>12.8f}".format("Blackman-Harris", sesgo_ax1_bh, var_ax1_bh))
print("{:<18} | {:>10.6f} | {:>12.8f}".format("Hann", sesgo_ax1_hann, var_ax1_hann))

print("\n")
print("========== Estimaciones para Señal x2 ==========")
print("{:<18} | {:>10} | {:>12}".format("Ventana", "Sesgo", "Varianza"))
print("-"*45)
print("{:<18} | {:>10.6f} | {:>12.8f}".format("Rectangular", sesgo_ax2_rect, var_ax2_rect))
print("{:<18} | {:>10.6f} | {:>12.8f}".format("Flattop", sesgo_ax2_flat, var_ax2_flat))
print("{:<18} | {:>10.6f} | {:>12.8f}".format("Blackman-Harris", sesgo_ax2_bh, var_ax2_bh))
print("{:<18} | {:>10.6f} | {:>12.8f}".format("Hann", sesgo_ax2_hann, var_ax2_hann))


#Tabla de estimación de frecuencia 

print("\n")
print("========== Estimación de Frecuencia para Señal x1 (SNR = 10 dB) ==========")
print("{:<18} | {:>12} | {:>12}".format("Ventana", "Sesgo ", "Varianza "))
print("-"*46)
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Rectangular", sesgo_fx1_rect, var_fx1_rect))
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Flattop", sesgo_fx1_flat, var_fx1_flat))
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Blackman-Harris", sesgo_fx1_bh, var_fx1_bh))
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Hann", sesgo_fx1_hann, var_fx1_hann))

print("\n")
print("========== Estimación de Frecuencia para Señal x2 (SNR = 3 dB) ==========")
print("{:<18} | {:>12} | {:>12}".format("Ventana", "Sesgo ", "Varianza "))
print("-"*46)
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Rectangular", sesgo_fx2_rect, var_fx2_rect))
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Flattop", sesgo_fx2_flat, var_fx2_flat))
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Blackman-Harris", sesgo_fx2_bh, var_fx2_bh))
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Hann", sesgo_fx2_hann, var_fx2_hann))



# --------------------------- Ploteos estimadores de amplitud --------------------------- #


plt.figure (2)

plt.subplot (2, 1, 1)
plt.hist (ax1_rect, bins=15, alpha=0.4, label='Rectangular')
plt.hist (ax1_flat, bins=15, alpha=0.7, label='Flattop')
plt.hist (ax1_bh, bins=15, alpha=0.4, label='Blackman-Harris')
plt.hist (ax1_hann, bins=15, alpha=0.2, label='Hann')
plt.axvline (x=a0, linestyle='--', label='Amplitud esperada')
plt.title ('Histograma de estimadores de amplitud para señal con ruido SNR = 3 dB')
plt.ylabel ('Realizaciones (R)')
plt.xlabel ('Amplitud estimada')
plt.grid (True)
plt.legend ()

plt.subplot (2, 1, 2)
plt.hist (ax2_rect, bins=15, alpha=0.4, label='Rectangular')
plt.hist (ax2_flat, bins=15, alpha=0.7, label='Flattop')
plt.hist (ax2_bh, bins=15, alpha=0.4, label='Blackman-Harris')
plt.hist (ax2_hann, bins=15, alpha=0.2, label='Hann')
plt.axvline (x=0, linestyle='--', color='red', label='Amplitud esperada')
plt.title ('Histograma de estimadores de amplitud para señal con ruido SNR = 10 dB')
plt.ylabel ('Realizaciones (R)')
plt.xlabel ('Amplitud estimada')
plt.grid (True)
plt.legend ()

plt.tight_layout ()
plt.show ()

# --------------------------- Ploteos de estimadores de frecuencia --------------------------- #


plt.figure (4)

plt.subplot (2, 1, 1)
plt.hist (fx1_rect, bins=15, alpha=0.4, label='Rectangular')
plt.hist (fx1_flat, bins=15, alpha=0.7, label='Flattop')
plt.hist (fx1_bh, bins=15, alpha=0.4, label='Blackman-Harris')
plt.hist (fx1_hann, bins=15, alpha=0.2, label='Hann')
plt.title ('Histograma de estimadores de frecuencia para señal x1 con ruido SNR = 3db')
plt.ylabel ('Realizaciones (R)')
plt.xlabel ('Frecuencia estimada (Hz)')
plt.legend ()
plt.grid (True)

plt.subplot (2, 1, 2)
plt.hist (fx2_rect, bins=15,alpha=0.4, label='Rectangular')
plt.hist (fx2_flat, bins=15, alpha=0.7, label='Flattop')
plt.hist (fx2_bh, bins=15,alpha=0.4, label='Blackman-Harris')
plt.hist (fx2_hann, bins=15, alpha=0.2, label='Hann')
plt.title ('Histograma de estimadores de frecuencia para señal x2 con ruido SNR = 10db')
plt.ylabel ('Realizaciones (R)')
plt.xlabel ('Frecuencia estimada (Hz)')
plt.legend ()
plt.grid (True)

plt.tight_layout ()
plt.show ()











