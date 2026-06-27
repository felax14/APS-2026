import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================
# FILTRO ANALOGICO T(s) = (s^2 + 4) / (s^2 + 2*sqrt(2)*s + 4)
# ceros en +-2j -> radio 2 (sobre el eje jw)
# polos en -sqrt(2)+-sqrt(2)j -> radio 2
# ============================================================
b = [1, 0, 4] # s^2 + 4
a = [1, 2*np.sqrt(2), 4] # s^2 + 2*sqrt(2)*s + 4

# --- Respuesta en frecuencia analogica ---
w = np.logspace(-2, 2, 1000) # 0.01 a 100 rad/s
w, h = signal.freqs(b, a, w)

plt.figure()
plt.semilogx(w, 20*np.log10(abs(h)))
plt.title("Magnitud analogica |H(jw)|")
plt.xlabel("w [rad/s]")
plt.ylabel("dB")
plt.grid(True, which="both", ls=":")

# --- Polos y ceros en el plano s ---
z, p, k = signal.tf2zpk(b, a)
print("ceros (s):", z, " radio:", np.abs(z))
print("polos (s):", p, " radio:", np.abs(p))

plt.figure(figsize=(6, 6))
ax = plt.gca()
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos')
plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title("Polos y ceros (plano s)")
plt.xlabel("sigma")
plt.ylabel("jw")
plt.axis('equal')
plt.grid(True)
plt.legend()

# ============================================================
# FILTRO DIGITAL por transformada bilineal: s = 2*fs*(z-1)/(z+1)
# ============================================================
fs = 3 # <-- PROBAR distintos valores: 0.5, 1, 3
num, den = signal.bilinear(b, a, fs)

# --- Respuesta en frecuencia digital (theta en [0, pi] rad/muestra) ---
wz, hz = signal.freqz(num, den, worN=1000)

plt.figure()
plt.plot(wz/np.pi, 20*np.log10(abs(hz)))
plt.title("Magnitud digital |H|, fs = %g" % fs)
plt.xlabel(r"theta / \pi [rad/muestra]") # Corregido con r"..." para LaTeX
plt.ylabel("dB")
plt.grid(True, which="both", ls=":")

# --- Polos y ceros en el plano z ---
zz, pz, kz = signal.tf2zpk(num, den)
print("ceros (z): radio", np.abs(zz), " angulo[rad]", np.angle(zz))
print("polos (z): radio", np.abs(pz), " angulo[rad]", np.angle(pz))

plt.figure(figsize=(6, 6))
ax = plt.gca()
plt.plot(np.real(pz), np.imag(pz), 'x', markersize=10, label='Polos')
plt.plot(np.real(zz), np.imag(zz), 'o', markersize=10, fillstyle='none', label='Ceros')
unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
ax.add_patch(unit_circle)
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.title("Polos y ceros (plano z)")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.grid(True)
plt.legend()

plt.show()
