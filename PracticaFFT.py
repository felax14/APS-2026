import numpy as np
import matplotlib.pyplot as plt

# Parámetros
N = 8
n = np.arange(N)
x = 4 + 3 * np.sin(np.pi * n / 2)  # Señal: x[n] = 4 + 3*sin(pi*n/2)

# DFT de N puntos
X = np.fft.fft(x, n=N)
k = np.arange(N)

# Impresión de resultados en consola
print("x[n]:", np.round(x, 4))
print("\nX[k] (complejo):")
for kk in range(N):
    print(f"  k={kk:2d}: {X[kk]:.3f}")

# Gráficos
fig, axes = plt.subplots(3, 1, figsize=(8, 7))

# Gráfico de la señal en el tiempo
axes[0].stem(n, x)
axes[0].set_title("Señal en tiempo: x[n] = 4 + 3*sin(pi*n/2)")
axes[0].set_xlabel("n")
axes[0].set_ylabel("x[n]")
axes[0].grid(True)

# Gráfico de Magnitud de la DFT
axes[1].stem(k, np.abs(X))
axes[1].set_title("DFT (N=8): Magnitud |X[k]|")
axes[1].set_xlabel("k")
axes[1].set_ylabel("|X[k]|")
axes[1].grid(True)

# Gráfico de Fase de la DFT
axes[2].stem(k, np.angle(X))
axes[2].set_title("DFT (N=8): Fase angulo(X[k]) [rad]")
axes[2].set_xlabel("k")
axes[2].set_ylabel("Fase [rad]")
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Verificacion: X[0]=32, X[2]=-12j, X[6]=+12j, resto ~0

import numpy as np
import matplotlib.pyplot as plt

# 1. Definir la señal original (L=3)
x_org = np.array([1.0, 1.0, 1.0])
L = len(x_org)

# 2. DFT con N = 3
N1 = 3
X1 = np.fft.fft(x_org, n=N1)
# Frecuencias de 0 a 2pi
freq1 = np.linspace(0, 2 * np.pi, N1, endpoint=False)

# 3. DFT con Zero-padding (N = 64)
N2 = 64
X2 = np.fft.fft(x_org, n=N2)
freq2 = np.linspace(0, 2 * np.pi, N2, endpoint=False)

# --- Creación de los gráficos ---
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Gráfico 1: Comparación de Magnitud
axes[0].stem(freq1, np.abs(X1), linefmt='C3-', markerfmt='C3o', label=f'DFT N={N1}')
axes[0].plot(freq2, np.abs(X2), 'C0--', alpha=0.6, label='Envolvente (DTFT continua)')
axes[0].stem(freq2, np.abs(X2), linefmt='C0-', markerfmt='C0.', label=f'DFT con Zero-Padding (N={N2})', basefmt=" ")

axes[0].set_title("Efecto del Zero-Padding en la Magnitud")
axes[0].set_xlabel("Frecuencia [rad/muestra]")
axes[0].set_ylabel("Magnitud |X[k]|")
axes[0].legend()
axes[0].grid(True)

# Gráfico 2: Fase (Solo donde hay magnitud significativa)
# Filtramos el ruido numérico para que la fase no se vea caótica en los ceros
X2_filtrada = X2.copy()
X2_filtrada[np.abs(X2) < 1e-10] = 0 
axes[1].plot(freq2, np.angle(X2_filtrada), color='green')
axes[1].set_title("Fase de la DFT (N=64)")
axes[1].set_xlabel("Frecuencia [rad/muestra]")
axes[1].set_ylabel("Fase [rad]")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Verificación por consola
print("Resultados N=3:")
for i, val in enumerate(X1):
    print(f"  X[{i}] = {np.round(val, 2)}")

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del ejercicio
fs = 1000        # Frecuencia de muestreo
N = 100          # Número de muestras
df = fs / N      # Resolución espectral (10 Hz)
n = np.arange(N)
f_bins = np.arange(N) * df  # Vector de frecuencias para el eje x

# a) Señal x1 con f = 200 Hz (Múltiplo de df)
f1 = 200
x1 = np.cos(2 * np.pi * f1 * n / fs)
X1_mag = np.abs(np.fft.fft(x1))

# b) Señal x2 con f = 205 Hz (NO es múltiplo de df)
f2 = 205
x2 = np.cos(2 * np.pi * f2 * n / fs)
X2_mag = np.abs(np.fft.fft(x2))

# --- Gráficos ---
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Gráfico para x1 (200 Hz)
axes[0].stem(f_bins[:N//2], X1_mag[:N//2], linefmt='C0-', markerfmt='C0o', basefmt=" ")
axes[0].set_title(f"Espectro x1: f = {f1} Hz (Múltiplo exacto de $\Delta f = {df}$ Hz)")
axes[0].set_ylabel("Magnitud")
axes[0].set_xlabel("Frecuencia [Hz]")
axes[0].grid(True, alpha=0.3)

# Gráfico para x2 (205 Hz)
axes[1].stem(f_bins[:N//2], X2_mag[:N//2], linefmt='C1-', markerfmt='C1o', basefmt=" ")
axes[1].set_title(f"Espectro x2: f = {f2} Hz (Desparramo Espectral / Leakage)")
axes[1].set_ylabel("Magnitud")
axes[1].set_xlabel("Frecuencia [Hz]")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Análisis de picos por consola
print(f"Magnitud en 200Hz (Bin 20) para x1: {X1_mag[20]:.2f}")
print(f"Magnitud en 200Hz (Bin 20) para x2: {X2_mag[20]:.2f}")
print(f"Magnitud en 210Hz (Bin 21) para x2: {X2_mag[21]:.2f}")