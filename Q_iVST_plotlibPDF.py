import os
import re
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import iv, k0

start_time = time.time()

# Parametri da settare manualmente
m_run = 'P143'
temperatures = [50, 100, 150, 200, 250, 300, 350, 400, 450]  # in mK
power = '0'  # dBm
freq_range = '680-960'
num_pixels = 143

# Directory where the text files are saved (se è la stessa dello script lasciare così)
directory = '.'

# Nome file da prendere
file_names = [
    f"ANALYSIS_{m_run}_{temp}mK_{power}dBm_{freq_range}.txt" for temp in temperatures
]

# Dictionary to save data
data = {pixel: [] for pixel in range(1, num_pixels + 1)}
coupling_qfs = {pixel: None for pixel in range(1, num_pixels + 1)}
quality_factors = {pixel: [] for pixel in range(1, num_pixels + 1)}
errors = {pixel: {'QF': [], 'Qi': [], 'Qc': []} for pixel in range(1, num_pixels + 1)}

# Regex to extract the temperature value from file names
temperature_pattern = re.compile(r'(\d+)mK')

# Function to extract data from the text file
def extract_data_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    temperature_match = temperature_pattern.search(file_path)
    if not temperature_match:
        return
    temperature = int(temperature_match.group(1))
    
    # Find wanted data for each pixel
    pixel_blocks = content.split('#______PIXEL #N:')[1:]
    
    for block in pixel_blocks:
        lines = block.split('\n')
        # Use regex to extract the pixel number
        pixel_number_match = re.match(r'(\d+)', lines[0].strip())
        if pixel_number_match:
            pixel_number = int(pixel_number_match.group(1))
            for line in lines:
                if "INTERNAL_QF:" in line:
                    internal_qf = float(line.split(":")[1].strip())
                    data[pixel_number].append((temperature, internal_qf))
                elif "ERROR_QI:" in line:
                    error_qi = float(line.split(":")[1].strip())
                    errors[pixel_number]['Qi'].append(error_qi)
                elif "COUPLING_QF:" in line:
                    coupling_qf = float(line.split(":")[1].strip())
                    if temperature == min(temperatures):  # Store Q_c at the lowest temperature
                        coupling_qfs[pixel_number] = coupling_qf
                elif "ERROR_QC:" in line:
                    error_qc = float(line.split(":")[1].strip())
                    errors[pixel_number]['Qc'].append(error_qc)
                elif "QUALITY_FACTOR:" in line:
                    quality_factor = float(line.split(":")[1].strip())
                    quality_factors[pixel_number].append((temperature, quality_factor))
                elif "ERROR_QF:" in line:
                    error_qf = float(line.split(":")[1].strip())
                    errors[pixel_number]['QF'].append(error_qf)

# Extract data from all the files in the directory
for filename in file_names:
    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        extract_data_from_file(file_path)
    else:
        print(f"File {filename} not found!")

# Chiede all'utente se si vuole per tutti i pixel un valore fisso del Qc oppure di prendere in esame il valore proprio ricavato da 4 FIT (nel secondo caso mandare "n")
use_fixed_qc = input("Do you want to use the fixed Q_c value (1e5) for all pixels? (y/n): ").strip().lower() == 'y'
fixed_qc_value = 1e5


def y_formatter(x, pos):
    return f'{x:,.0f}'

# Costanti modello teorico
hbar = 6.582119569e-16  # reduced Planck's constant in eV*s
kB = 8.617333262145e-5  # Boltzmann constant in eV/K
N_0 = 1.73E10
Delta0 = 1.76 * 1.275 * kB
sigma_n = 100  # Normal conductivity Siemens (electrical conductance)/meter
omega = 2 * np.pi * 850e6
alpha = 0.051

# Temperature range
T_theoretical = np.linspace(0.001, 10, 1000000)  # from 0.001 K to 10 K

n_qp_1 = 2 * N_0 * np.sqrt(2 * np.pi * kB * T_theoretical * Delta0) * np.exp(-Delta0 / (kB * T_theoretical))
Delta = ((2 * N_0 * Delta0) + np.sqrt((4 * N_0 * N_0 * Delta0 * Delta0) - 8 * N_0 * n_qp_1 * Delta0)) / (4 * N_0)
sigma1 = sigma_n * ((4 * Delta) / (hbar * omega)) * np.exp(-Delta / (kB * T_theoretical)) * np.sinh((hbar * omega) / (2 * kB * T_theoretical)) * k0((hbar * omega) / (2 * kB * T_theoretical))
sigma2 = sigma_n * ((np.pi * Delta) / (hbar * omega)) * (1 - (np.sqrt((2 * np.pi * kB * T_theoretical) / Delta) * np.exp(-Delta / (kB * T_theoretical))) - 2 * np.exp(-Delta / (kB * T_theoretical)) * np.exp(-(hbar * omega) / (2 * kB * T_theoretical)) * iv(0, (hbar * omega) / (2 * kB * T_theoretical)))
Q_i_theoretical = (1 / alpha) * (sigma2 / sigma1)


pdf_filename = f'QualityFactors_Vs_Temperature_{m_run}.pdf'
with PdfPages(pdf_filename) as pdf:

    for pixel, values in data.items():
        if not values:
            continue
        

        values.sort()
        temperatures, internal_qfs = zip(*values)
        
        plt.figure()
        plt.errorbar(temperatures, internal_qfs, yerr=errors[pixel]['Qi'], marker='o', color='purple', label='$Q_i$', elinewidth=1.9)

        # Se è stato girato il vecchio 4FIT.py va commentata la riga sopra e sostituita con questa sotto
        #plt.plot(temperatures, internal_qfs, marker='o', color='purple', label='$Q_i$')

        plt.plot(T_theoretical * 1000, Q_i_theoretical, color='b', label='$Q_i$ (Th.)')  # T_theoretical in mK
        
        # Determine Q_c to use
        Q_c = fixed_qc_value if use_fixed_qc else coupling_qfs[pixel]
        Q_c_label = '$Q_c$ fixed' if use_fixed_qc else '$Q_c$'
        # Il valore di Qc è estratto per ogni singolo pixel al valore minimo di temperatura perché è quello il valore che mi interessa includere nel calcolo del Quality factor totale
        Q_theoretical = (Q_i_theoretical * Q_c) / (Q_i_theoretical + Q_c)
        
        plt.axhline(Q_c, color='g', label=Q_c_label)
        plt.plot(T_theoretical * 1000, Q_theoretical, color='r', linestyle='--', label='$Q$ (Th.)')
        

        if quality_factors[pixel]:
            quality_factors[pixel].sort()
            _, qf_values = zip(*quality_factors[pixel])
            plt.errorbar(temperatures, qf_values, yerr=errors[pixel]['QF'], color='black', marker='x', linestyle=':', label='$Q$')

            #Se è stato girato il vecchio 4FIT.py va commentata la riga sopra e sostituita con questa sotto
            #plt.plot(temperatures, qf_values, color='black', marker='x', linestyle=':', label='$Q$')
        
        plt.title(f'Pixel {pixel}')
        plt.xlabel('Temperature [mK]')
        plt.ylabel('Quality Factors')
        plt.grid(True)
        plt.yscale('log')
        plt.tight_layout()
        plt.gca().yaxis.set_major_formatter(FuncFormatter(y_formatter))
        plt.xlim(0, 520)
        plt.ylim(1e2, 1e7)
        plt.legend()
        

        pdf.savefig()
        plt.close()

        print(f"Pixel {pixel} saved correctly in the PDF.")

end_time = time.time()

print(f"Process completed. The PDF has been saved as '{pdf_filename}'")
total_time = end_time - start_time
print(f"Total time: {total_time:.2f} seconds")
