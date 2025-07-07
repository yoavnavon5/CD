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

def plot_255_275_heatmap(d, vmin = -2, vmax = 2):
    """
    Plot a heatmap depicting the CD values in the range of 255-275 nm for different peptide concentrations and the two long DNA sequences (HOR7 and AIM18).
    Args:
        d: a dictionary containing the 255-275 nm CD values of both sequences for different peptide concentrations
        vmin: The minimal CD value to start the heatmap palette from
        vmax: The maximal CD value to start the heatmap palette from
    """
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 1.5

    #Plotting a separate heatmap for each promoter (HOR7 and AIM18)
    for key in d.keys():
        fig, ax = plt.subplots(figsize = (10, 3))
        sns.heatmap(ax = ax, data = d[key], annot = False, cmap = 'RdBu_r', vmin = vmin, vmax = vmax, fmt = '.2f')
        ax.set_title(key)
        fig.savefig('Heatmap_255_275_' + key + '_FOR_UPLOAD.pdf', format = 'pdf')
    return

def plot_255_275_with_buffer_scatter(d):
    """
    Plot each peptide concentration as a line connecting dots, whose colors correspond to the wavelength at the range of 255-275 nm: AIM18 CD values in the x-axis and HOR7 CD values in the y-axis.
    Args:
        d: a dictionary containing the 255-275 nm CD values of both sequences for different peptide concentrations
    """

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 1.5

    #Defining the dataframes corresponding to the sequences
    first_label = list(d.keys())[0]; second_label = list(d.keys())[1]
    first_df = d[first_label]; second_df = d[second_label]
    column_order = list(first_df.copy().columns[:-1])
    column_order.insert(0, first_df.copy().columns[-1])
    column_order = list(reversed(list(column_order)))
    first_df_ordered = first_df.copy()[column_order]
    second_df_ordered = second_df.copy()[column_order]

    #Defining a combined dataframe for easy plotting
    combined_df = pd.DataFrame(columns = [first_label, second_label, 'Concentration', 'Wavelength'])
    first_label_vector = []; second_label_vector = []; concentrations = []; wavelengths = []
    for col in first_df_ordered.columns:
        for wavelength in first_df_ordered.index:
            if col in ['buffer', '120uM', '150uM', '180uM', '210uM']:
                concentrations.append(col)
                wavelengths.append(wavelength)
                first_label_vector.append(first_df_ordered[col][wavelength])
                second_label_vector.append(second_df_ordered[col][wavelength])
    combined_df[first_label] = first_label_vector
    combined_df[second_label] = second_label_vector
    combined_df['Concentration'] = concentrations
    combined_df['Wavelength'] = wavelengths

    #Plotting the lines and scatters
    fig, ax = plt.subplots(figsize = (9, 6))
    sns.lineplot(ax = ax, data = combined_df, x = first_label, y = second_label, hue = concentrations, palette = 'Greys_r', linewidth = 0.5, zorder = 0)
    sns.scatterplot(ax = ax, data = combined_df, x = first_label, y = second_label, hue = wavelengths, palette = 'mako', edgecolors = 'k', linewidth = 0.25, s = 50, zorder = 1)
    ax.set_xlabel(first_label)
    ax.set_ylabel(second_label)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xticks(ticks = [-1.5, 0, 1.5], labels = [-1.5, 0, 1.5])
    ax.set_yticks(ticks = [-1.5, 0, 1.5], labels = [-1.5, 0, 1.5])
    fig.savefig(second_label + '_vs_' + first_label + '_scatter_255-275_wavelengths_as_colors_FOR_UPLOAD.pdf', format = 'pdf')

def plot_peak_analysis(files, peptide):
    """
    Plot peak analysis for a set of files.
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
    
    d_255_275 = {}
    d_255_275_with_buffer = {}
    start_255 = 255; end_275 = 275
    region_titles = ['Region 1 (205-230 nm)', 'Region 2 (230-260 nm)', 'Region 3 (257-310 nm)']
    
    # Collect peak data
    peak_data = {1: {}, 2: {}, 3: {}}  # region: {dna: {concentration: (wavelength, value)}}
    
    #Run over the different DNA sequences and their corresponding files
    for dna, dna_files in dna_groups.items():
        for file_path in dna_files:
            try:
                df = pd.read_csv(file_path)
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                wavelength_col = [col for col in df.columns if 'wave' in col.lower()][0]
                df = df[(df[wavelength_col] >= 205) & (df[wavelength_col] <= 320)]

                concentrations = [col for col in df.columns if col not in [wavelength_col, 'buffer']]
                buffer_conc_vec = [col for col in df.columns if col in ['buffer']]
                buffer_conc = buffer_conc_vec[0]
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
                    
                    #Define the dataframe which will contain the data for 255-275 nm plots (scatter and heatmap)
                    if (region_idx == 3) and ('_long' in dna):
                        df_255_275 = pd.DataFrame(columns = concentrations, index = list(np.linspace(start_255, end_275, int(np.abs(end_275 - start_255)) + 1).astype(int)))
                    
                    smoothed_buffer = savgol_filter(df[buffer_conc], window_length = 15, polyorder = 3)

                    # Now for each concentration, find peak location but measure value at min_conc_peak_wavelength
                    for i_conc, conc in enumerate(concentrations):
                        smoothed = savgol_filter(df[conc], window_length=15, polyorder=3)
                        region_mask = (df[wavelength_col] >= start) & (df[wavelength_col] <= end)
                        region_data = smoothed[region_mask]
                        region_wavelengths = df[wavelength_col][region_mask]

                        #Filling the earlier defined dataframe with the 255-275 nm data
                        if (region_idx == 3) and ('_long' in dna):
                            region_mask_255_275 = (df[wavelength_col] >= start_255) & (df[wavelength_col] <= end_275)
                            region_data_255_275 = list(reversed(list(smoothed[region_mask_255_275])))
                            region_data_buffer_255_275 = list(reversed(list(smoothed_buffer[region_mask_255_275])))
                            df_255_275[conc] = region_data_255_275
                            if i_conc == len(concentrations) - 1:
                                df_255_275_with_buffer = df_255_275.copy()
                                df_255_275_with_buffer['buffer'] = region_data_buffer_255_275
                                d_255_275[dna] = df_255_275
                                d_255_275_with_buffer[dna] = df_255_275_with_buffer

                        
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
    
    #Generating the 255-275 nm heatmaps
    fig_255_275 = plot_255_275_heatmap(d_255_275)

    #Generating the 255-275 nm scatters
    if len(d_255_275_with_buffer.keys()) > 0:
        fig_255_275_with_buffer = plot_255_275_with_buffer_scatter(d_255_275_with_buffer)
    
    #Assigning each DNA sequence the peak wavelength of the third region, minimal peptide concentration (5 uM)
    wavelengths_region3_minconc = []
    chosen_dnas = ['hor7_long', 'aim18_long']
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

            buffer_dnas = ['hor7_long', 'aim18_long']
            buffer_colors = plt.cm.RdBu_r(np.linspace(0, 1, len(buffer_dnas)))
            curr_dna_index = buffer_dnas.index(dna)
            curr_color = buffer_colors[curr_dna_index]

            #Plotting the normalized CD values vs. peptide concentration
            ax_norm.plot(concs, norm_values, marker='o', markeredgecolor = 'k', linewidth = 1.5, markeredgewidth = 0.25, label=dna, color = curr_color)
            dict_colors_legend[dna] = curr_color
            i_dna += 1
        
        # Set titles and labels        
        ax_norm.set_title(f'Normalized Peak Value vs Conc. - {region_titles[region_idx-1]}')
        ax_norm.set_xlabel('Concentration (μM)')
        ax_norm.set_ylabel('Normalized Peak Value')
        ax_norm.grid(True, alpha=0.3)
        ax_norm.legend(dict_colors_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def main():
    # Get all files - SPECIFY THE PATHS TO THE DATA FILES HERE
    all_files = glob.glob('./Downloads/peptides_DNA_buffer_FOR_UPLOAD/peptides_DNA_buffer/*/long_DNA_*peptide_substracted*.csv') #SPECIFY THE PATHS TO THE DATA FILES HERE
    
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
        fig1.savefig(f'output/peak_analysis_{peptide}_long_DNAs_FOR_UPLOAD.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig1)
    
    print("\nPDF files have been saved in the 'output' directory")

if __name__ == "__main__":
    main() 