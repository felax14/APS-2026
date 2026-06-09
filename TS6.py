import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

#%% PUNTO A
# Coeficientes del sistema a)
b = [1, 1, 1, 1]  # Numerador: z^3 + z^2 + z + 1
a = [1, 0, 0, 0]  # Denominador: z^3 + z^2 + z + 1

# 1. Calcular la respuesta en frecuencia (usamos solo el primer coeficiente para freqz)
w, h = signal.freqz(b, [1], worN=8000)

# 2. Calcular polos y ceros con el denominador completo
ceros, polos, ganancia = signal.tf2zpk(b, a)

# --- Gráficos de Respuesta en Frecuencia ---
plt.figure(figsize=(10, 9))

# Magnitud
plt.subplot(3, 1, 1)
plt.plot(w, np.abs(h), 'b', linewidth=2)
plt.title('Respuesta en Frecuencia - Sistema a)')
plt.ylabel('Módulo |T(e^{j\omega})|')
plt.grid(True)
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', 'π/4', 'π/2', '3π/4', 'π'])

# Fase
plt.subplot(3, 1, 2)
plt.plot(w, np.angle(h), 'r', linewidth=2)
plt.ylabel('Fase (Radianes)')
plt.grid(True)
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', 'π/4', 'π/2', '3π/4', 'π'])

# --- Gráfico de Polos y Ceros ---
plt.subplot(3, 1, 3)

# Dibujar la circunferencia unitaria
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', label='Circunferencia Unitaria')

# Graficar ceros (o) y polos (x)
plt.scatter(np.real(ceros), np.imag(ceros), s=100, marker='o', facecolors='none', edgecolors='g', linewidth=2, label='Ceros')
plt.scatter(np.real(polos), np.imag(polos), s=120, marker='x', color='r', linewidth=2, label='Polos (Orden 3)')

# Configuración del plano Z
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title('Diagrama de Polos y Ceros (Plano Z - Corregido)')
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginaria')

# Forzar que el gráfico esté centrado en el origen para ver bien el polo
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axis('equal')  
plt.grid(True)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

#%% PUNTO B 
# Coeficientes del sistema b)
b = [1, 1, 1, 1, 1]  # Numerador
a = [1, 0, 0, 0, 0]  # Denominador

# 1. Calcular la respuesta en frecuencia
w, h = signal.freqz(b, a, worN=8000)


# 2. Calcular polos y ceros
ceros, polos, ganancia = signal.tf2zpk(b, a)

# --- Gráficos de Respuesta en Frecuencia ---
plt.figure(figsize=(10, 8))

# Magnitud
plt.subplot(3, 1, 1)
plt.plot(w, np.abs(h), 'b', linewidth=2)
plt.title('Respuesta en Frecuencia - Sistema b)')
plt.ylabel('Módulo |T(e^{j\omega})|')
plt.grid(True)
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', 'π/4', 'π/2', '3π/4', 'π'])

# Fase
plt.subplot(3, 1, 2)
plt.plot(w, np.angle(h), 'r', linewidth=2)
plt.ylabel('Fase (Radianes)')
plt.grid(True)
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', 'π/4', 'π/2', '3π/4', 'π'])

# --- Gráfico de Polos y Ceros ---
plt.subplot(3, 1, 3)

# Dibujar la circunferencia unitaria
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', label='Circunferencia Unitaria')

# Graficar ceros (o) y polos (x)
plt.scatter(np.real(ceros), np.imag(ceros), s=80, marker='o', facecolors='none', edgecolors='g', linewidth=2, label='Ceros')
plt.scatter(np.real(polos), np.imag(polos), s=80, marker='x', color='r', linewidth=2, label='Polos')

# Configuración del plano Z
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title('Diagrama de Polos y Ceros (Plano Z)')
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginaria')
plt.axis('equal')  # Mantener proporciones circulares
plt.grid(True)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
#%% PUNTO C
# Coeficientes del sistema c) y(n) = x(n) - x(n-1)
b = [1, -1]
a = [1,  0]

# 1. Calcular la respuesta en frecuencia
w, h = signal.freqz(b, a, worN=8000)

# 2. Calcular polos y ceros
ceros, polos, ganancia = signal.tf2zpk(b, a)

# --- Gráficos de Respuesta en Frecuencia ---
plt.figure(figsize=(10, 8))

# Magnitud
plt.subplot(3, 1, 1)
plt.plot(w, np.abs(h), 'b', linewidth=2)
plt.title('Respuesta en Frecuencia - Sistema c)')
plt.ylabel('Módulo |T(e^{j\omega})|')
plt.grid(True)
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', 'π/4', 'π/2', '3π/4', 'π'])

# Fase
plt.subplot(3, 1, 2)
plt.plot(w, np.angle(h), 'r', linewidth=2)
plt.ylabel('Fase (Radianes)')
plt.grid(True)
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', 'π/4', 'π/2', '3π/4', 'π'])

# --- Gráfico de Polos y Ceros ---
plt.subplot(3, 1, 3)

# Dibujar la circunferencia unitaria
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', label='Circunferencia Unitaria')

# Graficar ceros (o) y polos (x)
plt.scatter(np.real(ceros), np.imag(ceros), s=80, marker='o', facecolors='none', edgecolors='g', linewidth=2, label='Ceros')
plt.scatter(np.real(polos), np.imag(polos), s=80, marker='x', color='r', linewidth=2, label='Polos')

# Configuración del plano Z
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title('Diagrama de Polos y Ceros (Plano Z)')
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginaria')
plt.axis('equal')  # Mantener proporciones circulares
plt.grid(True)
plt.legend(loc='upper right')


plt.tight_layout()
plt.show()
#%% PUNTO D
# Coeficientes del sistema c) y(n) = x(n) - x(n-1)
b = [1, 0, -1]
a = [1, 0,  0]

# 1. Calcular la respuesta en frecuencia
w, h = signal.freqz(b, a, worN=8000)

# 2. Calcular polos y ceros
ceros, polos, ganancia = signal.tf2zpk(b, a)

# --- Gráficos de Respuesta en Frecuencia ---
plt.figure(figsize=(10, 8))

# Magnitud
plt.subplot(3, 1, 1)
plt.plot(w, np.abs(h), 'b', linewidth=2)
plt.title('Respuesta en Frecuencia - Sistema d)')
plt.ylabel('Módulo |T(e^{j\omega})|')
plt.grid(True)
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', 'π/4', 'π/2', '3π/4', 'π'])

# Fase
plt.subplot(3, 1, 2)
plt.plot(w, np.angle(h), 'r', linewidth=2)
plt.ylabel('Fase (Radianes)')
plt.grid(True)
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', 'π/4', 'π/2', '3π/4', 'π'])

# --- Gráfico de Polos y Ceros ---
plt.subplot(3, 1, 3)

# Dibujar la circunferencia unitaria
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', label='Circunferencia Unitaria')

# Graficar ceros (o) y polos (x)
plt.scatter(np.real(ceros), np.imag(ceros), s=80, marker='o', facecolors='none', edgecolors='g', linewidth=2, label='Ceros')
plt.scatter(np.real(polos), np.imag(polos), s=80, marker='x', color='r', linewidth=2, label='Polos')

# Configuración del plano Z
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title('Diagrama de Polos y Ceros (Plano Z)')
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginaria')
plt.axis('equal')  # Mantener proporciones circulares
plt.grid(True)
plt.legend(loc='upper right')


plt.tight_layout()
plt.show()

