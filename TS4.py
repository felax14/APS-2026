import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.signal.windows as window
import scipy.stats as st


def señal (tt, frec, amp, SNR, R, fase=0, v_medio=0):

    N = len(tt)
    s = amp*np.sin(2*np.pi*frec*tt + fase) + v_medio
    if SNR == None:
      return s

    P_signal = np.var(s)
    P_noise  = P_signal / (10**(SNR/10))
    ruido_mat = np.random.normal (0, np.sqrt(P_noise), size=(N,R))
    return s + ruido_mat


#%% PARAMETROS 

fs = 1000
N = 1000
df = fs/N
R = 200
ff = np.arange (N) * df
ff_zp = fs * np.arange (10*N) / (10*N) #Zero Padding 


tt = np.arange (N) / fs                       
tt_col = tt.reshape (N, 1)                 
tt_mat = np.tile (tt_col, (1, R))            


frec_rand = np.random.uniform (-2, 2, R)     
frec = (N/4 + frec_rand) * df              
frec_fila = frec.reshape (1, R)               
frec_mat = np.tile (frec_fila, (N, 1))   

#%% DECLARACION DE VENTANAS 

# Modelo de señal (SNR en dB)

amp_0 = np.sqrt (2)
x1 = señal (tt = tt_mat, frec = frec_mat, amp = amp_0, SNR = 3, R = R)  # (NxR)
x2 = señal (tt = tt_mat, frec = frec_mat, amp = amp_0, SNR = 10, R = R) # (NxR)

# Generación de ventanas
w_rect = np.ones ((N, 1))                                  # Ventana rectangular (en realidad está implícita)
w_flat = window.flattop (N, sym=False).reshape(-1,1)       # Ventana Flattop
w_bh = window.blackmanharris (N, sym=False).reshape(-1,1)  # Ventana Blackman-Harris
w_hann = window.hann (N, sym=False).reshape(-1,1)          # Ventana Hann    

#%% VENTANEO Y FFTS 

# Ventaneo y FFT de la señal x1 (SNR = 3dB)

x1_rect = x1 * w_rect
x1_flat = x1 * w_flat
x1_bh   = x1 * w_bh
x1_hann = x1 * w_hann

X1_rect = (1/N) * fft(x1_rect, axis=0)
X1_flat = (1/N) * fft(x1_flat, axis=0)
X1_bh   = (1/N) * fft(x1_bh,   axis=0)
X1_hann = (1/N) * fft(x1_hann, axis=0)

# Con Zero Padding
X1_rect_zp = (1/N) * fft(x1_rect, n=10*N, axis=0)
X1_flat_zp = (1/N) * fft(x1_flat, n=10*N, axis=0)
X1_bh_zp   = (1/N) * fft(x1_bh, n=10*N,   axis=0)
X1_hann_zp = (1/N) * fft(x1_hann, n=10*N, axis=0)


# Ventaneo y FFT de la señal x2 (SNR = 10dB) 

x2_rect = x2 * w_rect
x2_flat = x2 * w_flat
x2_bh   = x2 * w_bh
x2_hann = x2 * w_hann

X2_rect = (1/N) * fft(x2_rect, axis=0)
X2_flat = (1/N) * fft(x2_flat, axis=0)
X2_bh   = (1/N) * fft(x2_bh,   axis=0)
X2_hann = (1/N) * fft(x2_hann, axis=0)

# Con Zero Padding
X2_rect_zp = (1/N) * fft(x2_rect, n=10*N, axis=0)
X2_flat_zp = (1/N) * fft(x2_flat, n=10*N, axis=0)
X2_bh_zp   = (1/N) * fft(x2_bh, n=10*N,   axis=0)
X2_hann_zp = (1/N) * fft(x2_hann, n=10*N, axis=0)

#%% GRAFICO DE ESPECTROS DE LAS SENALES 

#  Gráfico de PSD para señal x2 (SNR = 10 dB)

plt.figure (1)

plt.subplot (4, 1, 1)
plt.plot (ff, 10*np.log10(np.abs(X1_rect)**2), linewidth=0.6, alpha=0.6)
plt.title ('PSD - Ventana Rectangular (SNR = 3 dB)')
plt.ylabel ('[dB]')
plt.xlim (0, fs/2)
plt.grid (True)

plt.subplot (4, 1, 2)
plt.plot (ff, 10*np.log10(np.abs(X1_flat)**2), linewidth=0.6, alpha=0.6)
plt.title ('PSD - Ventana Flattop (SNR = 3 dB)')
plt.ylabel ('[dB]')
plt.xlim (0, fs/2)
plt.grid (True)

plt.subplot (4, 1, 3)
plt.plot (ff, 10*np.log10(np.abs(X1_bh)**2), linewidth=0.6, alpha=0.6)
plt.title ('PSD - Ventana Blackman-Harris (SNR = 3 dB)')
plt.ylabel ('[dB]')
plt.xlim (0, fs/2)
plt.grid (True)

plt.subplot (4, 1, 4)
plt.plot (ff, 10*np.log10(np.abs(X1_hann)**2), linewidth=0.6, alpha=0.6)
plt.title ('PSD - Ventana Hann (SNR = 3 dB)')
plt.xlabel ('Frecuencia [Hz]')
plt.ylabel ('[dB]')
plt.xlim (0, fs/2)
plt.grid (True)

plt.tight_layout ()
plt.show ()

#  Gráfico de PSD para señal x2 (SNR = 10 dB)  

plt.figure(4)

plt.subplot(4, 1, 1)
plt.plot(ff, 10*np.log10(np.abs(X2_rect)**2), linewidth=0.6, alpha=0.6)
plt.title('PSD - Ventana Rectangular (SNR = 10 dB)')
plt.ylabel('[dB]')
plt.xlim(0, fs/2)
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(ff, 10*np.log10(np.abs(X2_flat)**2), linewidth=0.6, alpha=0.6)
plt.title('PSD - Ventana Flattop (SNR = 10 dB)')
plt.ylabel('[dB]')
plt.xlim(0, fs/2)
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(ff, 10*np.log10(np.abs(X2_bh)**2), linewidth=0.6, alpha=0.6)
plt.title('PSD - Ventana Blackman-Harris (SNR = 10 dB)')
plt.ylabel('[dB]')
plt.xlim(0, fs/2)
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(ff, 10*np.log10(np.abs(X2_hann)**2), linewidth=0.6, alpha=0.6)
plt.title('PSD - Ventana Hann (SNR = 10 dB)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('[dB]')
plt.xlim(0, fs/2)
plt.grid(True)

plt.tight_layout()
plt.show()

#%% ESTIMADORES EN AMPLITUD Y FRECUENCIA 

#  Estimadores de amplitud de la señal x1 (SNR = 3dB)  


ax1_rect = 2*np.max(np.abs(X1_rect), axis=0) / np.mean(w_rect)
ax1_flat = 2*np.max(np.abs(X1_flat), axis=0) / np.mean(w_flat)
ax1_bh   = 2*np.max(np.abs(X1_bh),   axis=0) / np.mean(w_bh)
ax1_hann = 2*np.max(np.abs(X1_hann), axis=0) / np.mean(w_hann)

sesgo_ax1_rect = np.mean (ax1_rect) - amp_0 
sesgo_ax1_flat = np.mean (ax1_flat) - amp_0
sesgo_ax1_bh   = np.mean (ax1_bh)   - amp_0
sesgo_ax1_hann = np.mean (ax1_hann) - amp_0

var_ax1_rect = np.var (ax1_rect) 
var_ax1_flat = np.var (ax1_flat) 
var_ax1_bh   = np.var (ax1_bh)
var_ax1_hann = np.var (ax1_hann)


#  Estimadores de amplitud de la señal x2 (SNR = 10dB) 


ax2_rect = 2*np.max(np.abs(X2_rect), axis=0) / np.mean(w_rect)
ax2_flat = 2*np.max(np.abs(X2_flat), axis=0) / np.mean(w_flat)
ax2_bh   = 2*np.max(np.abs(X2_bh),   axis=0) / np.mean(w_bh)
ax2_hann = 2*np.max(np.abs(X2_hann), axis=0) / np.mean(w_hann)

sesgo_ax2_rect = np.mean (ax2_rect) - amp_0
sesgo_ax2_flat = np.mean (ax2_flat) - amp_0
sesgo_ax2_bh   = np.mean (ax2_bh)   - amp_0
sesgo_ax2_hann = np.mean (ax2_hann) - amp_0

var_ax2_rect = np.var (ax2_rect)
var_ax2_flat = np.var (ax2_flat)
var_ax2_bh   = np.var (ax2_bh)
var_ax2_hann = np.var (ax2_hann)


#  Estimadores de frecuencia de la señal x1 (SNR = 3dB) 


fx1_rect = np.argmax ((np.abs(X1_rect[0:N//2, :])), axis=0)
fx1_flat = np.argmax ((np.abs(X1_flat[0:N//2, :])), axis=0)
fx1_bh   = np.argmax ((np.abs(X1_bh[0:N//2, :])),   axis=0)
fx1_hann = np.argmax ((np.abs(X1_hann[0:N//2, :])), axis=0)

sesgo_fx1_rect = np.mean (fx1_rect - frec_mat[N//4, :]) # ahora mi valor de frecuencia conocido ya no está fijo, debo barrer todas las realizaciones
sesgo_fx1_flat = np.mean (fx1_flat - frec_mat[N//4, :])
sesgo_fx1_bh   = np.mean (fx1_bh   - frec_mat[N//4, :])
sesgo_fx1_hann = np.mean (fx1_hann - frec_mat[N//4, :])

var_fx1_rect = np.var (fx1_rect)
var_fx1_flat = np.var (fx1_flat)
var_fx1_bh   = np.var (fx1_bh)
var_fx1_hann = np.var (fx1_hann)


#  Estimadores de frecuencia de la señal x2 (SNR = 10dB)


fx2_rect = np.argmax ((np.abs(X2_rect[0:N//2, :])), axis=0)
fx2_flat = np.argmax ((np.abs(X2_flat[0:N//2, :])), axis=0)
fx2_bh   = np.argmax ((np.abs(X2_bh  [0:N//2, :])), axis=0)
fx2_hann = np.argmax ((np.abs(X2_hann[0:N//2, :])), axis=0)

sesgo_fx2_rect = np.mean (fx2_rect - N//4)
sesgo_fx2_flat = np.mean (fx2_flat - N//4)
sesgo_fx2_bh   = np.mean (fx2_bh   - N//4)
sesgo_fx2_hann = np.mean (fx2_hann - N//4)

var_fx2_rect = np.var (fx2_rect)
var_fx2_flat = np.var (fx2_flat)
var_fx2_bh   = np.var (fx2_bh)
var_fx2_hann = np.var (fx2_hann)


#  Tablas 

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


#  Tabla de estimación de frecuencia 

print("\n")
print("========== Estimación de Frecuencia para Señal x1 (SNR = 3 dB) ==========")
print("{:<18} | {:>12} | {:>12}".format("Ventana", "Sesgo ", "Varianza "))
print("-"*46)
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Rectangular", sesgo_fx1_rect, var_fx1_rect))
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Flattop", sesgo_fx1_flat, var_fx1_flat))
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Blackman-Harris", sesgo_fx1_bh, var_fx1_bh))
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Hann", sesgo_fx1_hann, var_fx1_hann))

print("\n")
print("========== Estimación de Frecuencia para Señal x2 (SNR = 10 dB) ==========")
print("{:<18} | {:>12} | {:>12}".format("Ventana", "Sesgo ", "Varianza "))
print("-"*46)
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Rectangular", sesgo_fx2_rect, var_fx2_rect))
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Flattop", sesgo_fx2_flat, var_fx2_flat))
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Blackman-Harris", sesgo_fx2_bh, var_fx2_bh))
print("{:<18} | {:>12.6f} | {:>12.8f}".format("Hann", sesgo_fx2_hann, var_fx2_hann))


#  Ploteos estimadores de amplitud 


plt.figure (2)

plt.subplot (2, 1, 1)
plt.hist (ax1_rect, bins=15, alpha=0.4, label='Rectangular')
plt.hist (ax1_flat, bins=15, alpha=0.7, label='Flattop')
plt.hist (ax1_bh, bins=15, alpha=0.4, label='Blackman-Harris')
plt.hist (ax1_hann, bins=15, alpha=0.2, label='Hann')
plt.axvline (x=amp_0, linestyle='--', label='Amplitud esperada')
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
plt.axvline (x=amp_0, linestyle='--', color='red', label='Amplitud esperada')
plt.title ('Histograma de estimadores de amplitud para señal con ruido SNR = 10 dB')
plt.ylabel ('Realizaciones (R)')
plt.xlabel ('Amplitud estimada')
plt.grid (True)
plt.legend ()

plt.tight_layout ()
plt.show ()

# Ploteos de estimadores de frecuencia


plt.figure (3)

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




