import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

b = [1, -1]
a = [1, 0]

w, h = signal.freqz(b, a, worN=2048)

fig, axs = plt.subplots(1, 3, figsize=(15,4))

def pi_ticks(ax):
    ticks = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    labels = ['0', 'π/4', 'π/2', '3π/4', 'π']
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(0, np.pi)

# Módulo
axs[0].plot(w, np.abs(h), color='steelblue')
axs[0].set_title('Módulo |T(e^jω)|')
axs[0].set_xlabel('ω [rad]')
pi_ticks(axs[0])
axs[0].grid(True)

# Fase
axs[1].plot(w, np.unwrap(np.angle(h)), color='darkorange')
axs[1].set_title('Fase')
axs[1].set_xlabel('ω [rad]')
axs[1].set_ylabel('rad')
pi_ticks(axs[1])
axs[1].grid(True)

# Polos y ceros
zeros, poles, k = signal.tf2zpk(b, a)
theta = np.linspace(0, 2*np.pi, 200)
axs[2].plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1)
axs[2].plot(zeros.real, zeros.imag, 'o', markersize=10,
            markerfacecolor='none', markeredgecolor='b', label='ceros')
axs[2].plot(poles.real, poles.imag, 'x', markersize=10,
            color='r', label='polos')
axs[2].axhline(0, color='gray', lw=0.5)
axs[2].axvline(0, color='gray', lw=0.5)
axs[2].set_xlim(-1.5,1.5); axs[2].set_ylim(-1.5,1.5)
axs[2].set_aspect('equal')
axs[2].set_title('Polos y ceros')
axs[2].legend()

plt.tight_layout()
plt.show()