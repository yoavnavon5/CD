import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from classify_names import classify_peptide, classify_dna
from pathlib import Path

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

def plot_titration_curves(files, dna):
    """
    Plot titration curves for a set of files, with one subplot per peptide.
    Args:
        files: List of file paths for the DNA
        dna: Name of the DNA being analyzed
    """
    # Group files by peptide
    peptide_groups = {}
    for file_path in files:
        dir_name = os.path.basename(os.path.dirname(file_path))
        peptide = classify_peptide(dir_name)
        if peptide:
            peptide_groups.setdefault(peptide, []).append(file_path)
    
    # Calculate grid dimensions
    n_peptide = len(peptide_groups)
    n_cols = min(3, n_peptide)  # Maximum 3 columns
    n_rows = (n_peptide + n_cols) // (n_cols * 2)  # Ceiling division
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each peptide
    for idx, (peptide, peptide_files) in enumerate(peptide_groups.items()):
        ax = axes[idx]
        
        for file_path in peptide_files:
            try:
                df = pd.read_csv(file_path)
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                wavelength_col = [col for col in df.columns if 'wave' in col.lower()][0]
                df = df[(df[wavelength_col] >= 210) & (df[wavelength_col] <= 320)]
                
                # Plot each concentration
                concentrations = [col for col in df.columns if col not in [wavelength_col, 'buffer']]
                concentrations.sort(key=lambda x: float(x.replace('uM', '')))
                cmap = list(plt.cm.Greys(np.linspace(0.05, 0.75, len(concentrations))))
                
                for i_conc, conc in enumerate(concentrations):
                    ax.plot(df[wavelength_col], df[conc], color = cmap[i_conc])
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('CD (mdeg)')
        ax.set_title(f'Peptide: {peptide}')
        ax.grid(True, alpha=0.3)
        ax.autoscale(enable=True, axis='y')  # Enable automatic y-axis scaling

    # Remove remaining empty subplots
    for idx in range(len(peptide_groups) + 1, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'Titration Curves for DNA: {dna}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def main():
    # Get all files - SPECIFY THE PATHS TO THE DATA FILES HERE
    all_files = glob.glob('./Downloads/peptides_DNA_buffer_FOR_UPLOAD/peptides_DNA_buffer/*/DNA_*peptide_substracted*.csv') #SPECIFY THE PATHS TO THE DATA FILES HERE
    
    # Group files by DNA
    dna_groups = {}
    for file_path in all_files:
        dir_name = os.path.basename(os.path.dirname(file_path))
        dna = classify_dna(dir_name)
        if dna:
            dna_groups.setdefault(dna, []).append(file_path)
    
    # Create output directory for PDFs
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Process each DNA
    for dna, files in dna_groups.items():
        print(f"\nProcessing DNA: {dna}")
        
        # Group files by peptide and handle duplicates
        peptide_files = {}
        for file_path in files:
            dir_name = os.path.basename(os.path.dirname(file_path))
            peptide = classify_peptide(dir_name)
            if peptide:
                if peptide in peptide_files:
                    # This is a duplicate, add suffix
                    existing_files = peptide_files[peptide]
                    suffix = chr(ord('a') + len(existing_files))
                    peptide_files[f"{peptide}_{suffix}"] = [file_path]
                else:
                    peptide_files[peptide] = [file_path]
        
        # Create figure with titration curves
        fig1 = plot_titration_curves(files, dna)
        # Save as PDF
        fig1.savefig(output_dir / f"titration_curves_FOR_UPLOAD_{dna}.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig1)

    print(f"\nPDF files have been saved to the {output_dir} directory. These PDFs can be opened and edited in Adobe Illustrator.")

if __name__ == "__main__":
    main() 