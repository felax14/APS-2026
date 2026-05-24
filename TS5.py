#%%
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
#%%
# =============================================================================
# 1. CARGA DE SEÑAL BIOMÉDICA (ECG SIN RUIDO)
# =============================================================================
# Cargo el diccionario completo desde el archivo .mat
ECG_contenido = sio.loadmat('ECG_TP4.mat')

# Señal de ECG cruda (una sola derivación)
ecg_one_lead = ECG_contenido['ecg_lead'] 

# Cargar la versión de la señal de ECG ya procesada (sin ruido)
ECG_sr = np.load('ecg_sin_ruido.npy')

# Defino Parametros 
fs_ECG = 1000                  # Frecuencia de muestreo (Hz)
N_ECG_sr = len(ECG_sr)         # Largo de la señal
df_ECG_sr = fs_ECG / N_ECG_sr  # Resolucion Espectral teórica (FFT completa)
nn_ECG_sr = np.arange(N_ECG_sr) # Vector de muestras 

# =============================================================================
# 2. ESTIMACIÓN ESPECTRAL VÍA MÉTODO WELCH (ECG SIN RUIDO)
# =============================================================================
promedios_ECG_sr = 12 
nperseg_ECG_sr = N_ECG_sr // promedios_ECG_sr 
nfft_ecg_pad = nperseg_ECG_sr * 10 # Zero-padding aplicado al ECG
ff_ECG_sr, per_ECG_sr = sig.welch(ECG_sr, fs=fs_ECG, window='hamming', scaling='density', nperseg=nperseg_ECG_sr, nfft=nfft_ecg_pad) 

per_ECG_sr_norm = per_ECG_sr / np.max(np.abs(per_ECG_sr)) # Normalizo la señal

per_ECG_sr_db = 10 * np.log10(per_ECG_sr_norm) # La convierto a dB 

# =============================================================================
# 3. CÁLCULO DE ANCHO DE BANDA (BW) POR ACUMULACIÓN DE POTENCIA (99% Central)
# =============================================================================
energia_acum_sr = np.cumsum(per_ECG_sr) 
energia_acum_sr_norm = energia_acum_sr / energia_acum_sr[-1] 

# Buscamos los índices que encierran el 99% central de la potencia
idx_inf = int(np.where(energia_acum_sr_norm >= 0.005)[0][0])
idx_sup = int(np.where(energia_acum_sr_norm >= 0.995)[0][0])

frec_inf_sr = ff_ECG_sr[idx_inf]
frec_corte_sr = ff_ECG_sr[idx_sup]
ancho_banda_ecg = frec_corte_sr - frec_inf_sr

print(f"--- Resultados ECG sin Ruido ---")
print(f"Frecuencia inferior (0.5%): {frec_inf_sr:.2f} Hz")
print(f"Frecuencia superior (99.5%): {frec_corte_sr:.2f} Hz")
print(f"Ancho de Banda (99%): {ancho_banda_ecg:.2f} Hz")

# =============================================================================
# 4. PLOT DE SEÑAL ECG Y SU PSD
# =============================================================================
plt.figure(2, figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(nn_ECG_sr, ECG_sr, color='darkmagenta')
plt.title("Señal de ECG sin ruido (Dominio del Tiempo)")
plt.ylabel("Amplitud")
plt.xlabel("Muestras")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(ff_ECG_sr, per_ECG_sr_db, color='magenta', label='PSD (Welch)')
plt.axvline(frec_inf_sr, linestyle=':', color='black', label=f'F_inf = {frec_inf_sr:.2f} Hz')
plt.axvline(frec_corte_sr, linestyle='--', color='darkmagenta', label=f'F_sup (Corte) = {frec_corte_sr:.2f} Hz')
plt.title(f"Densidad Espectral de Potencia (Ancho de Banda = {ancho_banda_ecg:.2f} Hz)")
plt.ylabel("Potencia espectral [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True)
plt.legend()
plt.xlim(0, 50) # Acotamos la vista a la zona de interés del ECG

plt.tight_layout()
plt.show()
#%%
# =============================================================================
# 1. CARGA DE SEÑAL BIOMÉDICA (ECG CON RUIDO)
# =============================================================================

# Cargar la versión de la señal de ECG ya procesada (con ruido)
ECG_cr = ecg_one_lead[670000:700000].ravel()

# Defino Parametros 
fs_ECG = 1000                  # Frecuencia de muestreo (Hz)
N_ECG_cr = len (ECG_cr)       # Largo de la señal
df_ECG_cr = fs_ECG / N_ECG_cr  # Resolucion Espectral teórica (FFT completa)
nn_ECG_cr = np.arange (N_ECG_cr) # Vector de muestras 

# =============================================================================
# 2. ESTIMACIÓN ESPECTRAL VÍA MÉTODO WELCH (ECG CON RUIDO)
# =============================================================================
promedios_ECG_cr = 15 
nperseg_ECG_cr = N_ECG_cr // promedios_ECG_cr 
nfft_ecg_pad = nperseg_ECG_cr * 10 # Zero-padding aplicado al ECG
ff_ECG_cr, per_ECG_cr = sig.welch(ECG_cr, fs=fs_ECG, window='hamming', scaling='density', nperseg=nperseg_ECG_cr, nfft=nfft_ecg_pad) 

per_ECG_cr_norm = per_ECG_cr / np.max(np.abs(per_ECG_cr)) # Normalizo la señal

per_ECG_cr_db = 10 * np.log10(per_ECG_cr_norm) # La convierto a dB 

# =============================================================================
# 3. CÁLCULO DE ANCHO DE BANDA (BW) POR ACUMULACIÓN DE POTENCIA (99% Central)
# =============================================================================
energia_acum_cr = np.cumsum(per_ECG_cr) 
energia_acum_cr_norm = energia_acum_cr / energia_acum_cr[-1] 

# Buscamos los índices que encierran el 99% central de la potencia
idx_inf = int(np.where(energia_acum_cr_norm >= 0.005)[0][0])
idx_sup = int(np.where(energia_acum_cr_norm >= 0.995)[0][0])

frec_inf_cr = ff_ECG_cr[idx_inf]
frec_corte_cr = ff_ECG_cr[idx_sup]
ancho_banda_ecg_cr = frec_corte_cr - frec_inf_cr

print(f"--- Resultados ECG con Ruido ---")
print(f"Frecuencia inferior (0.5%): {frec_inf_cr:.2f} Hz")
print(f"Frecuencia superior (99.5%): {frec_corte_cr:.2f} Hz")
print(f"Ancho de Banda (99%): {ancho_banda_ecg_cr:.2f} Hz")

# =============================================================================
# 4. PLOT DE SEÑAL ECG Y SU PSD
# =============================================================================
plt.figure(3, figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(nn_ECG_cr, ECG_cr, color='darkmagenta')
plt.title("Señal de ECG con ruido (Dominio del Tiempo)")
plt.ylabel("Amplitud")
plt.xlabel("Muestras")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(ff_ECG_cr, per_ECG_cr_db, color='magenta', label='PSD (Welch)')
plt.axvline(frec_inf_cr, linestyle=':', color='black', label=f'F_inf = {frec_inf_cr:.2f} Hz')
plt.axvline(frec_corte_cr, linestyle='--', color='darkmagenta', label=f'F_sup (Corte) = {frec_corte_cr:.2f} Hz')
plt.title(f"Densidad Espectral de Potencia (Ancho de Banda = {ancho_banda_ecg_cr:.2f} Hz)")
plt.ylabel("Potencia espectral [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True)
plt.legend()
plt.xlim(-1, 60) # Acotamos la vista a la zona de interés del ECG

plt.tight_layout()
plt.show()
#%% 
# =============================================================================
# 1. CARGA DE SEÑAL PPG (SIN RUIDO)
# =============================================================================

# Cargar la versión de la señal PPG (con ruido)
PPG_sr = np.load('ppg_sin_ruido.npy')

# Defino Parametros 
fs_PPG = 400                 # Frecuencia de muestreo (Hz)
N_PPG_sr = len (PPG_sr)       # Largo de la señal
df_PPG_sr = fs_PPG / N_PPG_sr  # Resolucion Espectral teórica (FFT completa)
nn_PPG_sr = np.arange (N_PPG_sr) # Vector de muestras 

# =============================================================================
# 2. ESTIMACIÓN ESPECTRAL VÍA MÉTODO WELCH (PPG SIN RUIDO)
# =============================================================================
promedios_PPG_sr = 12 
nperseg_PPG_sr = N_PPG_sr // promedios_PPG_sr 
nfft_ppg_pad = nperseg_PPG_sr * 10 # Zero-padding aplicado al PPG
ff_PPG_sr, per_PPG_sr = sig.welch(PPG_sr, fs=fs_PPG, window='hamming', scaling='density', nperseg=nperseg_PPG_sr, nfft=nfft_ppg_pad) 

per_PPG_sr_norm = per_PPG_sr / np.max(np.abs(per_PPG_sr)) # Normalizo la señal

per_PPG_sr_db = 10 * np.log10(per_PPG_sr_norm) # La convierto a dB   

# =============================================================================
# 3. CÁLCULO DE ANCHO DE BANDA (BW) POR ACUMULACIÓN DE POTENCIA (99% Central)
# =============================================================================
energia_acum_sr = np.cumsum(per_PPG_sr) 
energia_acum_sr_norm = energia_acum_sr / energia_acum_sr[-1] 

# Buscamos los índices que encierran el 99% central de la potencia
idx_inf = int(np.where(energia_acum_sr_norm >= 0.005)[0][0])
idx_sup = int(np.where(energia_acum_sr_norm >= 0.995)[0][0])

frec_inf_sr = ff_PPG_sr[idx_inf]
frec_corte_sr = ff_PPG_sr[idx_sup]
ancho_banda_ppg_sr = frec_corte_sr - frec_inf_sr

print(f"--- Resultados PPG sin Ruido ---")
print(f"Frecuencia inferior (0.5%): {frec_inf_sr:.2f} Hz")
print(f"Frecuencia superior (99.5%): {frec_corte_sr:.2f} Hz")
print(f"Ancho de Banda (99%): {ancho_banda_ppg_sr:.2f} Hz")

# =============================================================================
# 4. PLOT DE SEÑAL PPG Y SU PSD
# =============================================================================
plt.figure(4, figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(nn_PPG_sr, PPG_sr, color='darkmagenta')
plt.title("Señal de PPG sin ruido (Dominio del Tiempo)")
plt.ylabel("Amplitud")
plt.xlabel("Muestras")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(ff_PPG_sr, per_PPG_sr_db, color='magenta', label='PSD (Welch)')
plt.axvline(frec_inf_sr, linestyle=':', color='black', label=f'F_inf = {frec_inf_sr:.2f} Hz')
plt.axvline(frec_corte_sr, linestyle='--', color='darkmagenta', label=f'F_sup (Corte) = {frec_corte_sr:.2f} Hz')
plt.title(f"Densidad Espectral de Potencia (Ancho de Banda = {ancho_banda_ppg_sr:.2f} Hz)")
plt.ylabel("Potencia espectral [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True)
plt.legend()
plt.xlim(-1, 60) # Acotamos la vista a la zona de interés del ECG

plt.tight_layout()
plt.show()
#%% 
# =============================================================================
# 1. CARGA DE SEÑAL PPG (CON RUIDO)
# =============================================================================

# Cargar la versión de la señal PPG (con ruido)
PPG_cr = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1) # omite la cabecera, si existe

# Defino Parametros 
fs_PPG = 400                 # Frecuencia de muestreo (Hz)
N_PPG_cr = len (PPG_cr)     # Largo de la señal
df_PPG_cr = fs_PPG / N_PPG_cr  # Resolucion Espectral teórica (FFT completa)
nn_PPG_cr = np.arange (N_PPG_cr) # Vector de muestras 

# =============================================================================
# 2. ESTIMACIÓN ESPECTRAL VÍA MÉTODO WELCH (PPG CON RUIDO)
# =============================================================================
promedios_PPG_cr = 24
nperseg_PPG_cr = N_PPG_cr // promedios_PPG_cr 
nfft_ppg_pad = nperseg_PPG_cr * 10 # Zero-padding aplicado al ECG
ff_PPG_cr, per_PPG_cr = sig.welch(PPG_cr, fs=fs_PPG, window='hamming', scaling='density', nperseg=nperseg_PPG_cr, nfft=nfft_ppg_pad) 

per_PPG_cr_norm = per_PPG_cr / np.max(np.abs(per_PPG_cr)) # Normalizo la señal

per_PPG_cr_db = 10 * np.log10(per_PPG_cr_norm) # La convierto a dB   

# =============================================================================
# 3. CÁLCULO DE ANCHO DE BANDA (BW) POR ACUMULACIÓN DE POTENCIA (99% Central)
# =============================================================================
energia_acum_cr = np.cumsum(per_PPG_cr) 
energia_acum_cr_norm = energia_acum_cr / energia_acum_cr[-1] 

# Buscamos los índices que encierran el 99% central de la potencia
idx_inf = int(np.where(energia_acum_cr_norm >= 0.005)[0][0])
idx_sup = int(np.where(energia_acum_cr_norm >= 0.995)[0][0])

frec_inf_cr = ff_PPG_cr[idx_inf]
frec_corte_cr = ff_PPG_cr[idx_sup]
ancho_banda_ppg_cr = frec_corte_cr - frec_inf_cr

print(f"--- Resultados PPG con Ruido ---")
print(f"Frecuencia inferior (0.5%): {frec_inf_cr:.2f} Hz")
print(f"Frecuencia superior (99.5%): {frec_corte_cr:.2f} Hz")
print(f"Ancho de Banda (99%): {ancho_banda_ppg_cr:.2f} Hz")

# =============================================================================
# 4. PLOT DE SEÑAL PPG Y SU PSD
# =============================================================================
plt.figure(5, figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(nn_PPG_cr, PPG_cr, color='darkmagenta')
plt.title("Señal de PPG con ruido (Dominio del Tiempo)")
plt.ylabel("Amplitud")
plt.xlabel("Muestras")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(ff_PPG_cr, per_PPG_cr_db, color='magenta', label='PSD (Welch)')
plt.axvline(frec_inf_sr, linestyle=':', color='black', label=f'F_inf = {frec_inf_cr:.2f} Hz')
plt.axvline(frec_corte_sr, linestyle='--', color='darkmagenta', label=f'F_sup (Corte) = {frec_corte_cr:.2f} Hz')
plt.title(f"Densidad Espectral de Potencia (Ancho de Banda = {ancho_banda_ppg_cr:.2f} Hz)")
plt.ylabel("Potencia espectral [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True)
plt.legend()
plt.xlim(-1, 60) # Acotamos la vista a la zona de interés del ECG

plt.tight_layout()
plt.show()
#%% 
# =============================================================================
# 1. CARGO LAS SEÑALES DE AUDIO
# =============================================================================

fs_1, wav_data_1 = sio.wavfile.read ('prueba psd.wav') 
fs_2, wav_data_2 = sio.wavfile.read ('silbido.wav')
fs_3, wav_data_3 = sio.wavfile.read ('la cucaracha.wav')

# Defino Parametros 
N_1 = len(wav_data_1)
N_2 = len(wav_data_2)
N_3 = len(wav_data_3)

df_1 = fs_1 / N_1
df_2 = fs_2 / N_2
df_3 = fs_3 / N_3

tt_1 = np.arange (0, N_1/fs_1, 1/fs_1)
tt_2 = np.arange (0, N_2/fs_2, 1/fs_2)
tt_3 = np.arange (0, N_3/fs_3, 1/fs_3)

ff_1 = np.arange (N_1) * df_1
ff_2 = np.arange (N_2) * df_2
ff_3 = np.arange (N_3) * df_3

# =============================================================================
# 2.a. ESTIMACIÓN ESPECTRAL VÍA MÉTODO WELCH (prueba psd.wav)
# =============================================================================
# Utilizo método de Welch
promedios_audio_1 = 10 # defino la cantidad de bloques a promediar
nperseg_audio_1 = N_1 // promedios_audio_1
ff_1_welch, per_audio_1 = sig.welch (wav_data_1, nfft = 10*nperseg_audio_1, fs = fs_1, nperseg = nperseg_audio_1, window = 'hann')


per_audio_1_norm = per_audio_1 / np.max(np.abs(per_audio_1)) # Normalizo la señal

per_aduio_1_db = 10 * np.log10(per_audio_1_norm) # La convierto a dB   

# =============================================================================
# 2.b. ESTIMACIÓN ESPECTRAL VÍA MÉTODO WELCH (silbido.wav)
# =============================================================================
# Utilizo método de Welch
promedios_audio_2 = 10 # defino la cantidad de bloques a promediar
nperseg_audio_2 = N_2 // promedios_audio_2
ff_2_welch, per_audio_2 = sig.welch (wav_data_2, nfft = 10*nperseg_audio_2, fs = fs_2, nperseg = nperseg_audio_2, window = 'hann')


per_audio_2_norm = per_audio_2 / np.max(np.abs(per_audio_2)) # Normalizo la señal

per_aduio_2_db = 10 * np.log10(per_audio_2_norm) # La convierto a dB

# =============================================================================
# 2.c. ESTIMACIÓN ESPECTRAL VÍA MÉTODO WELCH (la cucaracha.wav)
# =============================================================================
# Utilizo método de Welch
promedios_audio_3 = 10 # defino la cantidad de bloques a promediar
nperseg_audio_3 = N_3 // promedios_audio_3
ff_3_welch, per_audio_3 = sig.welch (wav_data_3, nfft = 10*nperseg_audio_3, fs = fs_3, nperseg = nperseg_audio_3, window = 'hann')


per_audio_3_norm = per_audio_3 / np.max(np.abs(per_audio_3)) # Normalizo la señal

per_aduio_3_db = 10 * np.log10(per_audio_3_norm) # La convierto a dB

# =============================================================================
# 3.a CÁLCULO DE ANCHO DE BANDA (BW) (prueba psd.wav)
# =============================================================================
energia_acum_audio_1 = np.cumsum(per_audio_1) 
energia_acum_audio_1_norm = energia_acum_audio_1 / energia_acum_audio_1[-1] 

# Buscamos los índices que encierran el 99% central de la potencia
idx_inf = int(np.where(energia_acum_audio_1_norm >= 0.005)[0][0])
idx_sup = int(np.where(energia_acum_audio_1_norm >= 0.995)[0][0])

frec_inf_audio_1 = ff_1_welch[idx_inf]
frec_corte_audio_1 = ff_1_welch[idx_sup]
ancho_banda_audio_1 = frec_corte_audio_1 - frec_inf_audio_1

print(f"--- Resultados Audio (prueba psd.wav)  ---")
print(f"Frecuencia inferior (0.5%): {frec_inf_audio_1:.2f} Hz")
print(f"Frecuencia superior (99.5%): {frec_corte_audio_1:.2f} Hz")
print(f"Ancho de Banda (99%): {ancho_banda_audio_1:.2f} Hz")

# =============================================================================
# 3.b CÁLCULO DE ANCHO DE BANDA (BW) (silbido.wav)
# =============================================================================
energia_acum_audio_2 = np.cumsum(per_audio_2) 
energia_acum_audio_2_norm = energia_acum_audio_2 / energia_acum_audio_2[-1] 

# Buscamos los índices que encierran el 99% central de la potencia
idx_inf = int(np.where(energia_acum_audio_2_norm >= 0.005)[0][0])
idx_sup = int(np.where(energia_acum_audio_2_norm >= 0.995)[0][0])

frec_inf_audio_2 = ff_2_welch[idx_inf]
frec_corte_audio_2 = ff_2_welch[idx_sup]
ancho_banda_audio_2 = frec_corte_audio_2 - frec_inf_audio_2

print(f"--- Resultados Audio (silbido.wav) ---")
print(f"Frecuencia inferior (0.5%): {frec_inf_audio_2:.2f} Hz")
print(f"Frecuencia superior (99.5%): {frec_corte_audio_2:.2f} Hz")
print(f"Ancho de Banda (99%): {ancho_banda_audio_2:.2f} Hz")


# =============================================================================
# 3.c CÁLCULO DE ANCHO DE BANDA (BW) (la cucaracha.wav)
# =============================================================================
energia_acum_audio_3 = np.cumsum(per_audio_3) 
energia_acum_audio_3_norm = energia_acum_audio_3 / energia_acum_audio_3[-1] 

# Buscamos los índices que encierran el 99% central de la potencia
idx_inf = int(np.where(energia_acum_audio_3_norm >= 0.005)[0][0])
idx_sup = int(np.where(energia_acum_audio_3_norm >= 0.995)[0][0])

frec_inf_audio_3 = ff_3_welch[idx_inf]
frec_corte_audio_3 = ff_3_welch[idx_sup]
ancho_banda_audio_3 = frec_corte_audio_3 - frec_inf_audio_3

print(f"--- Resultados Audio (la cucaracha.wav) ---")
print(f"Frecuencia inferior (0.5%): {frec_inf_audio_3:.2f} Hz")
print(f"Frecuencia superior (99.5%): {frec_corte_audio_3:.2f} Hz")
print(f"Ancho de Banda (99%): {ancho_banda_audio_3:.2f} Hz")

# =============================================================================
# 4.a. PLOT DE LA  SEÑALES 1 (prueba psd.wav)
# =============================================================================
plt.figure(6, figsize=(10, 6))

# Subplot 1: Tiempo real con la señal original
plt.subplot(2, 1, 1)
plt.plot(tt_1, wav_data_1, color='darkmagenta') # X: tiempo, Y: datos de audio
plt.title("Señal de audio en el Dominio del Tiempo")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo [s]") # Si preferís muestras, cambiá tt_1 por np.arange(N_1) y esto a "Muestras"
plt.grid(True)

# Subplot 2: Frecuencia (PSD)
plt.subplot(2, 1, 2)
plt.plot(ff_1_welch, per_aduio_1_db, color='magenta', label='PSD (Welch)')
plt.axvline(frec_inf_audio_1, linestyle=':', color='black', label=f'F_inf = {frec_inf_audio_1:.2f} Hz')
# Corregido: frec_corte_cr cambiado por frec_corte_audio_1
plt.axvline(frec_corte_audio_1, linestyle='--', color='darkmagenta', label=f'F_sup (Corte) = {frec_corte_audio_1:.2f} Hz') 
plt.title(f"Densidad Espectral de Potencia (Ancho de Banda = {ancho_banda_audio_1:.2f} Hz)")
plt.ylabel("Potencia espectral [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# =============================================================================
# 4.b. PLOT DE LA SEÑAL 2 (silbido.wav)
# =============================================================================
plt.figure(7, figsize=(10, 6))

# Subplot 1: Tiempo
plt.subplot(2, 1, 1)
plt.plot(tt_2, wav_data_2, color='teal')
plt.title("Silbido - Dominio del Tiempo")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo [s]")
plt.grid(True)

# Subplot 2: Frecuencia (PSD)
plt.subplot(2, 1, 2)
plt.plot(ff_2_welch, per_aduio_2_db, color='darkcyan', label='PSD (Welch)')
plt.axvline(frec_inf_audio_2, linestyle=':', color='black', label=f'F_inf = {frec_inf_audio_2:.2f} Hz')
plt.axvline(frec_corte_audio_2, linestyle='--', color='red', label=f'F_sup (Corte) = {frec_corte_audio_2:.2f} Hz')
plt.title(f"Densidad Espectral de Potencia (Ancho de Banda = {ancho_banda_audio_2:.2f} Hz)")
plt.ylabel("Potencia espectral [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True)
plt.legend()
plt.xlim(0, fs_2 / 2) # Muestra todo el espectro útil hasta Nyquist

plt.tight_layout()
plt.show()

# =============================================================================
# 4.c. PLOT DE LA SEÑAL 3 (la cucaracha.wav)
# =============================================================================
plt.figure(8, figsize=(10, 6))

# Subplot 1: Tiempo
plt.subplot(2, 1, 1)
plt.plot(tt_3, wav_data_3, color='chocolate')
plt.title("La Cucaracha - Dominio del Tiempo")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo [s]")
plt.grid(True)

# Subplot 2: Frecuencia (PSD)
plt.subplot(2, 1, 2)
plt.plot(ff_3_welch, per_aduio_3_db, color='saddlebrown', label='PSD (Welch)')
plt.axvline(frec_inf_audio_3, linestyle=':', color='black', label=f'F_inf = {frec_inf_audio_3:.2f} Hz')
plt.axvline(frec_corte_audio_3, linestyle='--', color='red', label=f'F_sup (Corte) = {frec_corte_audio_3:.2f} Hz')
plt.title(f"Densidad Espectral de Potencia (Ancho de Banda = {ancho_banda_audio_3:.2f} Hz)")
plt.ylabel("Potencia espectral [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True)
plt.legend()
plt.xlim(0, fs_3 / 2) # Muestra todo el espectro útil hasta Nyquist

plt.tight_layout()
plt.show()





































