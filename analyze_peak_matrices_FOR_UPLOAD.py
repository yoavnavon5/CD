import matplotlib
matplotlib.use('Agg')  # Changed from TkAgg to Agg for non-interactive backend
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import glob
import os
from classify_names import classify_peptide, classify_dna
from pathlib import Path

# Set matplotlib to use vector-based rendering for Illustrator compatibility
plt.rcParams['pdf.fonttype'] = 42  # Editable text in Illustrator
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

def get_peak_values(file_path, target_conc=150):
    """
    Extracts peak values for a given file at target concentration and minimal concentration.
    Returns a dictionary with peak values for each region.
    """
    try:
        df = pd.read_csv(file_path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Find wavelength column
        wavelength_cols = [col for col in df.columns if 'wave' in col.lower()]
        if not wavelength_cols:
            raise ValueError(f"No wavelength column found in {file_path}")
        wavelength_col = wavelength_cols[0]
        
        df = df[(df[wavelength_col] >= 205) & (df[wavelength_col] <= 320)]
        
        # Get all concentration columns, including buffer
        concentrations = [col for col in df.columns if col not in [wavelength_col]]
        
        # Sort concentrations, treating buffer as 0
        def get_conc_value(col):
            if col.lower() == 'buffer':
                return 0
            return float(col.replace('uM', ''))
        
        concentrations.sort(key=get_conc_value)

        
        # Find minimal concentration (excluding buffer)
        min_conc = next((c for c in concentrations if c.lower() != 'buffer'), concentrations[0])
        
        # Find target concentration (or closest available)
        target_conc_str = f"{target_conc}uM"
        if target_conc_str not in concentrations:
            # Find closest concentration
            conc_values = [get_conc_value(c) for c in concentrations]
            closest_idx = np.argmin(np.abs(np.array(conc_values) - target_conc))
            target_conc_str = concentrations[closest_idx]
        
        # Define regions
        regions = {
            1: (205, 230, True),   # Region 1: 205-230 nm, find minima
            2: (230, 260, True),   # Region 2: 230-260 nm, find minima
            3: (257, 310, False)   # Region 3: 257-310 nm, find global maximum
        }
        
        peak_values = {}
        
        # Process each region
        for region_idx, (start, end, find_minima) in regions.items():
            try:
                # Get minimal concentration data
                min_conc_data = savgol_filter(df[min_conc], window_length=15, polyorder=3)
                region_mask = (df[wavelength_col] >= start) & (df[wavelength_col] <= end)
                region_data = min_conc_data[region_mask]
                region_wavelengths = df[wavelength_col][region_mask]
                
                if len(region_data) == 0:
                    print(f"Warning: No data points in region {region_idx} for {file_path}")
                    continue
                
                if region_idx == 3:
                    # For Region 3, find the global maximum
                    max_idx = np.argmax(region_data)
                    min_conc_peak_wavelength = region_wavelengths.iloc[max_idx]
                    min_conc_peak_value = region_data[max_idx]
                else:
                    # For Regions 1 and 2, find minima using find_peaks
                    if find_minima:
                        peaks, _ = find_peaks(-region_data, height=None, distance=10)
                    else:
                        peaks, _ = find_peaks(region_data, height=None, distance=10)
                    
                    if len(peaks) > 0:
                        min_conc_peak_wavelength = region_wavelengths.iloc[peaks[0]]
                        min_conc_peak_value = region_data[peaks[0]]
                    else:
                        print(f"Warning: No peaks found in region {region_idx} for {file_path}")
                        continue
                
                # Get target concentration data
                target_data = savgol_filter(df[target_conc_str], window_length=15, polyorder=3)
                target_region_data = target_data[region_mask]
                
                if region_idx == 3:
                    # For Region 3, find the global maximum
                    max_idx = np.argmax(target_region_data)
                    target_peak_wavelength = region_wavelengths.iloc[max_idx]
                    target_peak_value = target_region_data[max_idx]
                else:
                    # For Regions 1 and 2, find minima using find_peaks
                    if find_minima:
                        peaks, _ = find_peaks(-target_region_data, height=None, distance=10)
                    else:
                        peaks, _ = find_peaks(target_region_data, height=None, distance=10)
                    
                    if len(peaks) > 0:
                        target_peak_wavelength = region_wavelengths.iloc[peaks[0]]
                        target_peak_value = target_region_data[peaks[0]]
                    else:
                        print(f"Warning: No peaks found in region {region_idx} for {file_path}")
                        continue
                
                # Get all concentrations data for linear regression slope calculation
                peak_values_all_concentrations = []
                for curr_conc_str in concentrations: 
                    curr_all_data = savgol_filter(df[curr_conc_str], window_length=15, polyorder=3)
                    curr_all_region_data = curr_all_data[region_mask]
                    min_region_data = min_conc_data[region_mask]
                
                    if region_idx == 3:
                        # For Region 3, find the global maximum
                        max_idx_min = np.argmax(min_region_data)
                        curr_all_peak_value = curr_all_region_data[max_idx_min]
                    else:
                        # For Regions 1 and 2, find minima using find_peaks
                        if find_minima:
                            peaks, _ = find_peaks(-target_region_data, height=None, distance=10)
                        else:
                            peaks, _ = find_peaks(target_region_data, height=None, distance=10)
                    
                        if len(peaks) > 0:
                            curr_all_peak_value = curr_all_region_data[peaks[0]]
                        else:
                            print(f"Warning: No peaks found in region {region_idx} for {file_path}")
                            continue
                    peak_values_all_concentrations.append(curr_all_peak_value)
                numerical_concentrations = [get_conc_value(x) for x in concentrations]
                slope, intercept = np.polyfit(numerical_concentrations, peak_values_all_concentrations, 1)
                
                
                # Calculate normalized value
                normalized_value = target_peak_value / min_conc_peak_value if min_conc_peak_value != 0 else None
                
                peak_values[region_idx] = {
                    'min_conc_value': min_conc_peak_value,
                    'target_conc_value': target_peak_value,
                    'normalized_value': normalized_value,
                    'min_conc_wavelength': min_conc_peak_wavelength,
                    'target_conc_wavelength': target_peak_wavelength,
                    'slope': slope
                }
            except Exception as e:
                print(f"Error processing region {region_idx} in {file_path}: {str(e)}")
                continue
        
        return peak_values
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def get_dna_ordering(dnas, peptides, wavelength_matrices):
    """
    Order DNA sequences based on Region 3 peak wavelengths with VPVPV peptide.
    """
    if 'VPVPV' not in peptides:
        print("Warning: VPVPV peptide not found, using default DNA ordering")
        return dnas
    
    vpvpv_idx = peptides.index('VPVPV')
    dna_wavelengths = wavelength_matrices[3][:, vpvpv_idx]
    
    # Create list of (dna, wavelength) pairs, excluding NaN values
    dna_wavelength_pairs = [(dna, wavelength) for dna, wavelength in zip(dnas, dna_wavelengths) 
                           if not np.isnan(wavelength)]
    
    # Sort by wavelength
    sorted_pairs = sorted(dna_wavelength_pairs, key=lambda x: x[1])
    
    # Get ordered DNA list
    ordered_dnas = [pair[0] for pair in sorted_pairs]
    
    # Add any DNA sequences that had NaN values at the end
    nan_dnas = [dna for dna in dnas if dna not in ordered_dnas]
    ordered_dnas.extend(nan_dnas)
    
    return ordered_dnas

def get_peptide_ordering(dnas, peptides, normalized_matrices):
    """
    Order peptides based on the number of tested DNA regions (non-NaN values).
    Special handling for KRKRK and VPVPV.
    """
    # Count non-NaN values for each peptide across all DNA sequences
    peptide_counts = []
    for peptide_idx in range(len(peptides)):
        count = 0
        for region in [1, 2, 3]:
            count += np.sum(~np.isnan(normalized_matrices[region][:, peptide_idx]))
        peptide_counts.append((peptides[peptide_idx], count))
    
    # Sort by count in descending order
    sorted_pairs = sorted(peptide_counts, key=lambda x: x[1], reverse=True)
    
    # Get ordered peptide list
    ordered_peptides = [pair[0] for pair in sorted_pairs]
    
    # Special handling for KRKRK and VPVPV
    if 'KRKRK' in ordered_peptides and 'VPVPV' in ordered_peptides:
        krkrk_idx = ordered_peptides.index('KRKRK')
        vpvpv_idx = ordered_peptides.index('VPVPV')
        ordered_peptides[krkrk_idx], ordered_peptides[vpvpv_idx] = ordered_peptides[vpvpv_idx], ordered_peptides[krkrk_idx]
    
    return ordered_peptides

def normalize_dna_name(dna):
    """
    Normalize DNA names to handle control sequence variations.
    """
    # First remove 'DNA-' prefix if present
    dna = dna.replace('DNA-', '')
    # Then normalize all control variations to '40_1_cont'
    return dna.replace('40_1_cntlr', '40_1_cont')

def create_comparison_plot(normalized_matrices, wavelength_matrices, min_conc_matrices, peptides, dnas):
    """
    Create scatter plots comparing KRKRK and VPVPV normalized values for all three regions,
    with points colored by peak wavelengths and peak values.
    """
    if 'KRKRK' not in peptides or 'VPVPV' not in peptides:
        print("Warning: Either KRKRK or VPVPV not found in peptides")
        return None
    
    krkrk_idx = peptides.index('KRKRK')
    vpvpv_idx = peptides.index('VPVPV')

    #Defining the DNA sequence identifiers and position of third region peak, based on the buffer profiles analysis (see another .py file)
    buffer_dnas = ['viii', 'iii', 'i', 'ii', 'x', 'ix', 'vi', 'iv', 'vii', 'v'] #Extracted from buffer profiles analysis (see another .py file)
    buffer_third_peaks = [281, 272, 264, 265, 282, 281, 278, 273, 278, 273] #Extracted from buffer profiles analysis (see another .py file)
    ctrl_seq_index = 8 #Extracted from buffer profiles analysis (see another .py file)
    buffer_colors = plt.cm.RdBu_r(np.linspace(0, 1, len(buffer_dnas)))

    argsort_buffer_third_peaks = np.argsort(buffer_third_peaks)

    buffer_dnas_ordered_by_third = [buffer_dnas[i] for i in argsort_buffer_third_peaks]
    curr_buffer_dnas_ordered = buffer_dnas_ordered_by_third


    # Create a figure
    fig, axes = plt.subplots(figsize=(6.67, 6))
    fig.suptitle('KRKRK vs VPVPV Comparison by Region', fontsize=14)
    
    region_titles = ['Region 1 (205-230 nm)', 'Region 2 (230-260 nm)', 'Region 3 (257-310 nm)']
    
    for region_idx in range(3, 4): #Examining third region only
        # Get data for current region
        krkrk_values = normalized_matrices[region_idx][:, krkrk_idx]
        vpvpv_values = normalized_matrices[region_idx][:, vpvpv_idx]
        wavelength_values = wavelength_matrices[region_idx][:, krkrk_idx]
        peak_values = min_conc_matrices[region_idx][:, krkrk_idx]

        
        # Create dictionary to store data by normalized DNA name
        data_dict = {}
        
        # Collect data, handling DNA name variations
        for i, dna in enumerate(dnas):
            norm_dna = normalize_dna_name(dna)

            curr_color = buffer_colors[0]
            dna_in_buffers = False
            for buffer_dna in curr_buffer_dnas_ordered:
                if buffer_dna == norm_dna:
                    dna_in_buffers = True
                    curr_dna_index = curr_buffer_dnas_ordered.index(norm_dna)
                    curr_color = buffer_colors[curr_dna_index]
                    break
            if not dna_in_buffers:
                continue

            if not np.isnan(krkrk_values[i]) and not np.isnan(vpvpv_values[i]):
                if norm_dna not in data_dict:
                    data_dict[norm_dna] = {
                        'krkrk': krkrk_values[i],
                        'vpvpv': vpvpv_values[i],
                        'wavelength': wavelength_values[i],
                        'peak_value': peak_values[i],
                        'original_name': dna,
                        'color': curr_color
                    }
        
        # Add combined control point
        dna_ctrl_idx = dnas.index('viii') if 'viii' in dnas else None
        ctrl_idx = dnas.index('viii') if 'viii' in dnas else None
        
        if dna_ctrl_idx is not None and ctrl_idx is not None:
            if not np.isnan(krkrk_values[dna_ctrl_idx]) and not np.isnan(vpvpv_values[ctrl_idx]):
                data_dict['combined_ctrl'] = {
                    'krkrk': krkrk_values[dna_ctrl_idx],
                    'vpvpv': vpvpv_values[ctrl_idx],
                    'wavelength': wavelength_values[dna_ctrl_idx],
                    'peak_value': peak_values[dna_ctrl_idx],
                    'original_name': 'viii',
                    'color': buffer_colors[ctrl_seq_index]
                }

        # Plot with third peak value coloring
        scatter1 = axes.scatter(
            [-data['krkrk'] for data in data_dict.values()],
            [-data['vpvpv'] for data in data_dict.values()],
            c=[data['color'] for data in data_dict.values()],
            cmap='RdBu_r',
            edgecolors = 'k',
            linewidths = 0.6,
            s=200,
            alpha=0.7
        )
        
        # Add labels and titles
        axes.set_xlabel('-slope (KRKRK) [mDeg/uM, %]')
        axes.set_ylabel('-slope (VPVPV) [mDeg/uM, %]')
        axes.set_title(f'{region_titles[region_idx-1]}\nColored by Peak Value')
            
        # Add diagonal line
        min_val = min(axes.get_xlim()[0], axes.get_ylim()[0])
        max_val = max(axes.get_xlim()[1], axes.get_ylim()[1])
        axes.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

        #Add facecolors - sienna for hydrophobic affected region, plum for charged affected region
        facecolor_x = list(np.linspace(min_val, max_val, 100))
        facecolor_y = list(np.linspace(min_val, max_val, 100))
        axes.fill_between(facecolor_x, facecolor_x, np.max(facecolor_y), facecolor = 'sienna', alpha = 0.08, zorder = 0)
        axes.fill_between(facecolor_x, np.min(facecolor_y), facecolor_x, facecolor = 'plum', alpha = 0.15, zorder = 0)
            
        # Add DNA labels
        for dna, data in data_dict.items():
            axes.annotate(data['original_name'],
                        (-data['krkrk'], -data['vpvpv']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=20)
    
    plt.tight_layout()
    return fig

def create_peak_matrices():
    """
    Create matrices showing peak values for different peptide-DNA combinations.
    """
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Get all files - SPECIFY THE PATHS TO THE DATA FILES HERE
    all_files = glob.glob('./Downloads/peptides_DNA_buffer_FOR_UPLOAD/peptides_DNA_buffer/*/DNA_*peptide_substracted*.csv') #SPECIFY THE PATHS TO THE DATA FILES HERE
    
    # Collect unique peptides and DNA
    peptides = set()
    dnas = set()
    for file_path in all_files:
        dir_name = os.path.basename(os.path.dirname(file_path))
        peptide = classify_peptide(dir_name)
        dna = classify_dna(dir_name)
        if peptide and dna:
            peptides.add(peptide)
            dnas.add(dna)
    
    # Convert to lists
    peptides = list(peptides)
    dnas = list(dnas)
    
    # Initialize matrices
    normalized_matrices = {1: np.full((len(dnas), len(peptides)), np.nan),
                         2: np.full((len(dnas), len(peptides)), np.nan),
                         3: np.full((len(dnas), len(peptides)), np.nan)}
    
    min_conc_matrices = {1: np.full((len(dnas), len(peptides)), np.nan),
                       2: np.full((len(dnas), len(peptides)), np.nan),
                       3: np.full((len(dnas), len(peptides)), np.nan)}
    
    wavelength_matrices = {1: np.full((len(dnas), len(peptides)), np.nan),
                         2: np.full((len(dnas), len(peptides)), np.nan),
                         3: np.full((len(dnas), len(peptides)), np.nan)}
    
    target_wavelength_matrices = {1: np.full((len(dnas), len(peptides)), np.nan),
                                2: np.full((len(dnas), len(peptides)), np.nan),
                                3: np.full((len(dnas), len(peptides)), np.nan)}
    
    # Fill matrices
    for file_path in all_files:
        dir_name = os.path.basename(os.path.dirname(file_path))
        peptide = classify_peptide(dir_name)
        dna = classify_dna(dir_name)
        
        if peptide and dna:
            peak_values = get_peak_values(file_path)
            if peak_values:
                peptide_idx = peptides.index(peptide)
                dna_idx = dnas.index(dna)
                
                for region_idx in [1, 2, 3]:
                    if region_idx in peak_values:
                        normalized_matrices[region_idx][dna_idx, peptide_idx] = peak_values[region_idx]['slope']
                        min_conc_matrices[region_idx][dna_idx, peptide_idx] = peak_values[region_idx]['min_conc_value']
                        wavelength_matrices[region_idx][dna_idx, peptide_idx] = peak_values[region_idx]['min_conc_wavelength']
                        target_wavelength_matrices[region_idx][dna_idx, peptide_idx] = peak_values[region_idx]['target_conc_wavelength']
    
    # Get ordered lists
    ordered_dnas = get_dna_ordering(dnas, peptides, wavelength_matrices)
    ordered_peptides = get_peptide_ordering(dnas, peptides, normalized_matrices)
    
    # Create new ordered matrices
    ordered_normalized_matrices = {}
    ordered_min_conc_matrices = {}
    ordered_wavelength_matrices = {}
    ordered_target_wavelength_matrices = {}
    
    for region in [1, 2, 3]:
        # Create new matrices with ordered dimensions
        ordered_normalized_matrices[region] = np.full((len(dnas), len(peptides)), np.nan)
        ordered_min_conc_matrices[region] = np.full((len(dnas), len(peptides)), np.nan)
        ordered_wavelength_matrices[region] = np.full((len(dnas), len(peptides)), np.nan)
        ordered_target_wavelength_matrices[region] = np.full((len(dnas), len(peptides)), np.nan)
        
        # Fill matrices with ordered data
        for i, dna in enumerate(ordered_dnas):
            for j, peptide in enumerate(ordered_peptides):
                old_i = dnas.index(dna)
                old_j = peptides.index(peptide)
                ordered_normalized_matrices[region][i, j] = normalized_matrices[region][old_i, old_j]
                ordered_min_conc_matrices[region][i, j] = min_conc_matrices[region][old_i, old_j]
                ordered_wavelength_matrices[region][i, j] = wavelength_matrices[region][old_i, old_j]
                ordered_target_wavelength_matrices[region][i, j] = target_wavelength_matrices[region][old_i, old_j]
    
    # Create and save KRKRK vs VPVPV comparison plot
    comparison_fig = create_comparison_plot(normalized_matrices, wavelength_matrices, min_conc_matrices, peptides, dnas)
    if comparison_fig:
        comparison_fig.savefig(output_dir / 'krkrk_vpvpv_comparison.pdf', format='pdf', bbox_inches='tight')
        plt.close(comparison_fig)
    
    print(f"PDF files have been saved to {output_dir} directory. These PDFs can be opened and edited in Adobe Illustrator.")

#Calling the function plotting the peak analyses
if __name__ == "__main__":
    create_peak_matrices()