# CD
Circular Dichroism Data and Code.

Attached are .csv files (in the zipped directory), each corresponds to a measurement of a DNA sequence and an added peptide at different concentrations and different wavelengths of polarized light.
The columns refer to the concentrations (buffer relates to a measurement of a DNA only, the rest to an addition of a peptide), while the rows relate to the wavelength (in nm).
When downloading the directory, one should retain its structure for further analyses. Jupyter notebook (.ipynb) files are found in the sub-directories of the long HOR7 and AIM18 sequences and the short ii and viii sequences, for generating their respective CD spectra (panels 3A and 3F ,respectively). 

Attached are also the .py files for analyzing the CD data: the first file classifies the names of both DNA and peptides for further analyses; the second one plots the CD profiles of the buffer measurements (DNA alone, without a peptide); the third one analyzes the CD spectra peaks (extreme values) in relation to the wavelength and concentration; the fourth one classifies the data according to the peptides; and the fifth one classifies the data according to the DNA sequences. One may generate panels 3B, 3D, 3E, 3G, 3H and the supplementary ones via those files.
