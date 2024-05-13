import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

save_loc= './images'
sensor_data_loc= './sensor_data'

right = pd.read_csv(f'{sensor_data_loc}/right_01.csv')
left = pd.read_csv(f'{sensor_data_loc}/left_01.csv')
up = pd.read_csv(f'{sensor_data_loc}/up_01.csv')
down = pd.read_csv(f'{sensor_data_loc}/down_01.csv')

acce_names = right.columns[0:3]
gyro_names = right.columns[3:6]
x_axis = np.array(range(len(right)))
x_axis = (x_axis/float(len(right))) * 4  # Show 4 seconds on x-axis

movements = ['Right', 'Left', 'Up', 'Down']
components = ['x', 'y', 'z']

fs = 168



# Acceleration Over Time
for file, movement in zip([right, left, up, down], movements):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(3):
        ax.plot(x_axis, file[acce_names[i]], label=acce_names[i])
    ax.set_title(f"{movement} Acceleration Vs. Time")
    ax.grid(True)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Acceleration')
    ax.legend(title='Legend')
    plt.savefig(f'{save_loc}/acceleration_{movement.lower()}_00.png')
    plt.close()

# Gyroscope Over Time
for file, movement in zip([right, left, up, down], movements):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(3):
        ax.plot(x_axis, file[gyro_names[i]], label=gyro_names[i])
    ax.set_title(f"{movement} Rotation Vs. Time")
    ax.grid(True)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Rotation')
    ax.legend(title='Legend')
    plt.savefig(f'{save_loc}/gyro_{movement.lower()}_00.png')
    plt.close()


# Acceleration FFT
fig, axs = plt.subplots(1, 1, figsize=(8, 6))
plt.subplots_adjust(hspace=0.5, wspace=0.5)
ax = axs
ax.set_title("Acceleration FFT")
for movement, file in zip(movements, files):
    for col in acce_names:
        signal = file[col]
        signal = signal - np.mean(signal)
        fft_values = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), 1/fs)  
        ax.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_values[:len(fft_freq)//2]), label=f"{col} ({movement})")
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.legend(title='Legend')
        ax.set_xlim(0,20)
plt.savefig(f'{save_loc}/acceleration_FFT.png')

# Gyroscope FFT
fig, axs = plt.subplots(1, 1, figsize=(8, 6))
plt.subplots_adjust(hspace=0.5, wspace=0.5)
ax = axs
ax.set_title("Gyroscope FFT")
for movement, file in zip(movements, files):
    for col in gyro_names:
        signal = file[col]
        signal = signal - np.mean(signal)
        fft_values = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), 1/fs)  
        ax.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_values[:len(fft_freq)//2]), label=f"{col} ({movement})")
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.legend(title='Legend')
        ax.set_xlim(0,15)
plt.savefig(f'{save_loc}/gyroscope_FFT.png')




# Generate and save combined acceleration spectrogram for each movement
for movement, file in zip(movements, [up, down, left, right]):
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    
    # Reset total_signal for each movement
    total_signal = None
    
    for i, component in enumerate(components):
        signal = file[acce_names[i]]
        signal = signal - np.mean(signal)
        if total_signal is None:
            total_signal = signal ** 2
        else:
            total_signal += signal ** 2
            
    Pxx, freqs, bins, im = ax.specgram(total_signal, Fs=fs, NFFT=4 * fs, cmap='viridis')
    ax.set_title(f"{movement.capitalize()} Acceleration Spectrogram")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude (Accel)')
    plt.savefig(f'{save_loc}/acceleration_spectro_{movement}_01.png')
    plt.close()

# Generate and save combined gyroscope spectrogram for each movement
for movement, file in zip(movements, [up, down, left, right]):
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    
    # Reset total_signal for each movement
    total_signal = None
    
    for i, component in enumerate(components):
        signal = file[gyro_names[i]]
        signal = signal - np.mean(signal)
        if total_signal is None:
            total_signal = signal ** 2
        else:
            total_signal += signal ** 2
            
    Pxx, freqs, bins, im = ax.specgram(total_signal, Fs=fs, NFFT=4 * fs, cmap='viridis')
    ax.set_title(f"{movement.capitalize()} Gyroscope Spectrogram")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude (Rotation)')
    plt.savefig(f'{save_loc}/gyro_spectro_{movement}_01.png')
    plt.close()
