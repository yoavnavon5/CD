import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from classify_names import classify_dna

# Set matplotlib to use vector-based rendering
plt.rcParams['pdf.fonttype'] = 42  # This ensures text is editable in Illustrator
plt.rcParams['ps.fonttype'] = 42   # This ensures text is editable in Illustrator
plt.rcParams['svg.fonttype'] = 'none'  # This ensures text is editable in Illustrator
plt.rcParams['font.family'] = 'Arial'  # Use a standard font that's available in Illustrator
plt.rcParams['axes.linewidth'] = 1.0  # Thinner lines for better vector editing
plt.rcParams['lines.linewidth'] = 1.0  # Thinner lines for better vector editing
plt.rcParams['savefig.dpi'] = 300  # High resolution for any raster elements
plt.rcParams['savefig.format'] = 'pdf'  # Default to PDF format
plt.rcParams['savefig.bbox'] = 'tight'  # Tight bounding box
plt.rcParams['savefig.pad_inches'] = 0.1  # Small padding around the figure

def plot_buffer_profiles():
    """
    Plot all buffer profiles from the dataset:
    Curves are color-coded based on the location of their third peak (257-310 nm).
    """
    # Get all files - SPECIFY THE PATHS TO THE DATA FILES HERE
    all_files = glob.glob('./Downloads/peptides_DNA_buffer_FOR_UPLOAD/peptides_DNA_buffer/*/DNA_*peptide_substracted*.csv') #SPECIFY THE PATHS TO THE DATA FILES HERE
    
    # Create figure with two subplots side by side
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    # Dictionary to store buffer profiles by DNA
    buffer_profiles = {}
    
    first_locs = []
    third_locs = []
    dnas = []

    # Process each file
    for file_path in all_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # Find wavelength column
            wavelength_cols = [col for col in df.columns if 'wave' in col.lower()]
            if not wavelength_cols:
                print(f"No wavelength column found in {file_path}")
                continue
            wavelength_col = wavelength_cols[0]
            
            # Filter wavelength range
            df = df[(df[wavelength_col] >= 205) & (df[wavelength_col] <= 320)]
            
            # Get buffer data - try different possible column names
            buffer_cols = [col for col in df.columns if 'buffer' in col.lower()]
            if not buffer_cols:
                print(f"No buffer column found in {file_path}")
                continue
            buffer_col = buffer_cols[0]  # Use the first matching column
            
            # Get DNA name from directory
            dir_name = os.path.basename(os.path.dirname(file_path))
            dna = classify_dna(dir_name)
            if not dna:
                continue
            
            # Skip if it contains 'DNA' in the name
            if 'dna' in dna.lower():
                continue

            # Calculate standard deviation of the signal
            signal_std = df[buffer_col].std()
            
            #Calculate buffer profiles for DNA tested with the VPVPV peptide
            if 'VPVPV' not in file_path:
                continue
            
            # Find third peak location (in region 3)
            region3_mask = (df[wavelength_col] >= 257) & (df[wavelength_col] <= 310)
            region3_data = df.loc[region3_mask, buffer_col]
            region3_wavelengths = df.loc[region3_mask, wavelength_col]
            third_peak_loc = region3_wavelengths.iloc[region3_data.argmax()]
            third_locs.append(third_peak_loc)

            # Find first peak location (in region 3)
            region1_mask = (df[wavelength_col] >= 205) & (df[wavelength_col] <= 230)
            region1_data = df.loc[region1_mask, buffer_col]
            first_peak_loc = np.min(region1_data / signal_std)

            first_locs.append(first_peak_loc)
            dnas.append(dna)

            # Store normalized buffer profile and peak location
            buffer_profiles[dna] = {
                'wavelength': df[wavelength_col],
                'absorbance': df[buffer_col] / signal_std,  # Normalize by standard deviation
                'third_peak_loc': third_peak_loc,
                'first_peak_loc': first_peak_loc,
                'is_hor7_aim18': 'hor7' in dna.lower() or 'aim18' in dna.lower() or 'aim_18' in dna.lower()
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    if not buffer_profiles:
        print("No valid buffer profiles found to plot")
        return
    
    # # Sort profiles by third peak location
    sorted_all = sorted(buffer_profiles.items(), key = lambda x: x[1]['third_peak_loc'])
    
    # # Create color maps
    colors_all = plt.cm.RdBu_r(np.linspace(0, 1, len(sorted_all)))
    
    # Plot all profiles
    for (dna, data), color in zip(sorted_all, colors_all):
        ax1.plot(data['wavelength'], data['absorbance'], 
                label=f"{dna} (peak: {data['third_peak_loc']:.1f} nm)", 
                color=color, 
                alpha=0.7,
                linewidth=1.5,  # Slightly thicker lines for better visibility
                zorder=2)  # Ensure lines are above grid
    
    # Customize plots
    for ax, title in [(ax1, 'All Short DNA Profiles')]:
        ax.plot([x for x in range(200, 320)], [0 for x in range(200, 320)], '--k', zorder = 0, alpha = 0.5)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized Absorbance (by Std Dev)')
        ax.set_title(f'{title}\nColor indicates third peak location')
        ax.grid(True, alpha=0.3, zorder=1)  # Grid below lines
        ax.set_xlim(200, 320)
        ax.set_ylim(-3, 3)
        ax.set_xticks(list(np.linspace(200, 320, 13)))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
    
    # Add overall title
    fig.suptitle('Normalized Buffer Profiles Across All Samples (205-320 nm)', 
                 y=1.05, fontsize=14)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('buffer_profiles_normalized_std_colored_separated_FOR_UPLOAD.pdf', format='pdf', bbox_inches='tight')
    plt.close()


    #Sorting the colors according to the position/location of the third peak of each sequence
    argsort_third_locs = np.argsort(third_locs)
    argsort_argsort_third_locs = np.argsort(argsort_third_locs)
    sorted_colors_for_scatter = [colors_all[i] for i in argsort_argsort_third_locs]

    #Plotting the third peak location vs. the first peak value, colored by the third peak location
    fig, ax3 = plt.subplots(figsize=(7, 7))
    ax3.scatter(x = first_locs, y = third_locs, c = sorted_colors_for_scatter, label = dnas, s = 400, edgecolors = 'k', facecolors = 'skyblue', linewidth = 0.25)
    ax3.set_xlabel('First Peak Value [Normalized mDeg]', fontsize = 25)
    ax3.set_ylabel('Third Peak Location [nm]', fontsize = 25)
    ax3.tick_params(axis = 'both', labelsize = 20)
    fig.savefig('scatter_first_third_locs_colored_FOR_UPLOAD.pdf', format = 'pdf')
    plt.close()

#Calling the plotting function
if __name__ == "__main__":
    plot_buffer_profiles()