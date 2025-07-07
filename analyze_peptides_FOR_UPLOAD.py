import matplotlib
matplotlib.use('Agg')  # Changed from 'TkAgg' to 'Agg' for non-interactive plotting
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import savgol_filter, find_peaks
import glob
import os
from classify_names import classify_peptide, classify_dna

# Set PDF-specific parameters for Illustrator compatibility
plt.rcParams['pdf.fonttype'] = 42  # This ensures text is editable in Illustrator
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 1.5

def smooth_data(y, window_length=5, polyorder=3):
    """Apply Savitzky-Golay filter for smoothing with gentler parameters"""
    return savgol_filter(y, window_length, polyorder)

def find_peaks_in_region(wavelengths, data, region_min, region_max, find_minima=False):
    """Find peaks in a specific wavelength region"""
    # Filter data for the region
    mask = (wavelengths >= region_min) & (wavelengths <= region_max)
    region_wavelengths = wavelengths[mask]
    region_data = data[mask]
    
    # Smooth the data
    smoothed_data = smooth_data(region_data)
    
    # For minima, invert the data
    if find_minima:
        smoothed_data = -smoothed_data
    
    # Find peaks
    peaks, properties = find_peaks(smoothed_data)
    
    if find_minima:
        smoothed_data = -smoothed_data  # invert back
    
    return region_wavelengths[peaks], smoothed_data[peaks]

def plot_peak_analysis(files, peptide):
    """
    Plot peak analysis for a set of files, with a 3x3 grid of subplots.
    Args:
        files: List of file paths for the peptide
        peptide: Name of the peptide being analyzed
    """
    # Group files by DNA
    dna_groups = {}
    for file_path in files:
        dir_name = os.path.basename(os.path.dirname(file_path))
        dna = classify_dna(dir_name)
        if dna:
            dna_groups.setdefault(dna, []).append(file_path)
    
    # Create figure
    fig, axes = plt.subplots(figsize=(5.75, 5))
    fig.suptitle(f"Peak Analysis for Peptide: {peptide}", fontsize=20)
    
    # Define regions
    regions = {
        1: (205, 230, True),   # Region 1: 205-230 nm, find minima
        2: (230, 260, True),   # Region 2: 230-260 nm, find minima
        3: (257, 310, False)   # Region 3: 257-310 nm, find global maximum
    }
    
    region_titles = ['Region 1 (205-230 nm)', 'Region 2 (230-260 nm)', 'Region 3 (257-310 nm)']
    
    # Collect peak data
    peak_data = {1: {}, 2: {}, 3: {}}  # region: {dna: {concentration: (wavelength, value)}}
    
    for dna, dna_files in dna_groups.items():
        for file_path in dna_files:
            try:
                df = pd.read_csv(file_path)
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                wavelength_col = [col for col in df.columns if 'wave' in col.lower()][0]
                df = df[(df[wavelength_col] >= 205) & (df[wavelength_col] <= 320)]

                concentrations = [col for col in df.columns if col not in [wavelength_col, 'buffer']]
                concentrations.sort(key=lambda x: float(x.replace('uM', '')))
                
                # First, find the peak wavelength at minimal concentration
                min_conc = concentrations[0]
                min_conc_data = savgol_filter(df[min_conc], window_length=15, polyorder=3)
                
                for region_idx, (start, end, find_minima) in regions.items():
                    if dna not in peak_data[region_idx]:
                        peak_data[region_idx][dna] = {}
                    
                    # Find peak wavelength at minimal concentration
                    region_mask = (df[wavelength_col] >= start) & (df[wavelength_col] <= end)
                    region_data = min_conc_data[region_mask]
                    region_wavelengths = df[wavelength_col][region_mask]
                    
                    if region_idx == 3:
                        # For Region 3, find the global maximum
                        max_idx = np.argmax(region_data)
                        min_conc_peak_wavelength = region_wavelengths.iloc[max_idx]
                    else:
                        # For Regions 1 and 2, find minima using find_peaks
                        if find_minima:
                            peaks, _ = find_peaks(-region_data, height=None, distance=10)
                        else:
                            peaks, _ = find_peaks(region_data, height=None, distance=10)
                        
                        if len(peaks) > 0:
                            min_conc_peak_wavelength = region_wavelengths.iloc[peaks[0]]
                        else:
                            continue

                    # Now for each concentration, find peak location but measure value at min_conc_peak_wavelength
                    for conc in concentrations:
                        smoothed = savgol_filter(df[conc], window_length=15, polyorder=3)
                        region_mask = (df[wavelength_col] >= start) & (df[wavelength_col] <= end)
                        region_data = smoothed[region_mask]
                        region_wavelengths = df[wavelength_col][region_mask]
                        
                        if region_idx == 3:
                            # For Region 3, find the global maximum
                            max_idx = np.argmax(region_data)
                            peak_wavelength = region_wavelengths.iloc[max_idx]
                        else:
                            # For Regions 1 and 2, find minima using find_peaks
                            if find_minima:
                                peaks, _ = find_peaks(-region_data, height=None, distance=10)
                            else:
                                peaks, _ = find_peaks(region_data, height=None, distance=10)
                            
                            if len(peaks) > 0:
                                peak_wavelength = region_wavelengths.iloc[peaks[0]]
                            else:
                                continue
                        
                        # Measure value at min_conc_peak_wavelength
                        wavelength_idx = np.abs(df[wavelength_col] - min_conc_peak_wavelength).argmin()
                        peak_value = smoothed[wavelength_idx]
                        
                        conc_value = float(conc.replace('uM', ''))
                        peak_data[region_idx][dna][conc_value] = (peak_wavelength, peak_value)
            
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
    
    #Extracting the peak locations (wavelengths) of the third region 
    wavelengths_region3_minconc = []
    chosen_dnas = ['viii', 'iii', 'ii', 'i', 'vii', 'iv', 'v', 'vi', 'ix', 'x'] #Extracted from the buffer profiles analysis (see another .py file)
    for dna in peak_data[3]:
        if dna in chosen_dnas:
            wavelengths_region3_minconc.append(peak_data[3][dna][5.0][0])
    if len(wavelengths_region3_minconc) == 0:
        return fig
    dict_colors_legend = {}
    
    # Plot the data
    for region_idx in [3]:
        ax_norm = axes # Normalized peak value plot
        
        i_dna = 0
        for dna in peak_data[region_idx]:
            if dna not in chosen_dnas:
                continue
            concs = sorted(peak_data[region_idx][dna].keys())
            
            if not concs:
                continue
            
            # Extract data
            values = [peak_data[region_idx][dna][c][1] for c in concs]
            
            # Calculate normalized values
            min_conc = min(concs)
            min_value = peak_data[region_idx][dna][min_conc][1]
            norm_values = [v/min_value for v in values]

            #Defining the DNA sequence identifiers and position of third region peak, based on the buffer profiles analysis (see another .py file)
            buffer_dnas = ['viii', 'iii', 'i', 'ii', 'x', 'ix', 'vi', 'iv', 'vii', 'v'] #Extracted from buffer profiles analysis (see another .py file)
            buffer_third_peaks = [281, 272, 264, 265, 282, 281, 278, 273, 278, 273] #Extracted from buffer profiles analysis (see another .py file)
            buffer_colors = plt.cm.RdBu_r(np.linspace(0, 1, len(buffer_dnas)))

            argsort_buffer_third_peaks = np.argsort(buffer_third_peaks)
            buffer_dnas_ordered_by_third = [buffer_dnas[i] for i in argsort_buffer_third_peaks]
            curr_dna_index = buffer_dnas_ordered_by_third.index(dna)
            curr_color = buffer_colors[curr_dna_index]

            #Plotting the data
            ax_norm.plot(concs, norm_values, marker='o', markeredgecolor = 'k', linewidth = 1.5, markeredgewidth = 0.25, label=dna, color = curr_color)
            dict_colors_legend[dna + " " + str(np.round(sorted(buffer_third_peaks)[curr_dna_index], 2)) + " nm"] = curr_color
            i_dna += 1
        
        # Set titles and labels
        ax_norm.set_title(f'Normalized Peak Value vs Conc.\n{region_titles[region_idx-1]}')
        ax_norm.set_xlabel('Concentration (μM)')
        ax_norm.set_ylabel('Normalized Peak Value')
        ax_norm.set_ylim([0.6, 1.1])
        ax_norm.grid(True, alpha=0.3)
        ax_norm.legend(dict_colors_legend)
        
        
        ax_norm.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def main():
    # Get all files - SPECIFY THE PATHS TO THE DATA FILES HERE
    all_files = glob.glob('./Downloads/peptides_DNA_buffer_FOR_UPLOAD/peptides_DNA_buffer/*/DNA_*peptide_substracted*.csv') #SPECIFY THE PATHS TO THE DATA FILES HERE
    
    # Group files by peptide
    peptide_groups = {}
    for file_path in all_files:
        dir_name = os.path.basename(os.path.dirname(file_path))
        peptide = classify_peptide(dir_name)
        if peptide:
            peptide_groups.setdefault(peptide, []).append(file_path)
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Process each peptide
    for peptide, files in peptide_groups.items():
        print(f"\nProcessing peptide: {peptide}")
        
        # Create and save peak analysis figure
        fig1 = plot_peak_analysis(files, peptide)
        fig1.savefig(f'output/peak_analysis_{peptide}.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig1)
    
    print("\nPDF files have been saved in the 'output' directory")

if __name__ == "__main__":
    main() 