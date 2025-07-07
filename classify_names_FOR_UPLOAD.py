import re

def classify_peptide(dir_name):
    """
    Extract peptide name from directory name and add suffix for duplicate files.
    Pattern: 5 uppercase letters bordered by underscores (e.g., _VPVPV_) or ending the name (e.g., _VPVPV).
    For duplicate files of the same peptide, adds a suffix (a, b, etc.) to distinguish them.
    Args:
        dir_name (str): Directory name containing the peptide
    Returns:
        str or None: The 5-letter peptide sequence with optional suffix if found, None otherwise
    """
    peptide_match = re.search(r'_([A-Z]{5})(?:_|$)', dir_name)
    if peptide_match:
        # Get the base peptide name
        peptide = peptide_match.group(1)
        
        # Check if this is a duplicate file by looking for a suffix
        suffix_match = re.search(r'_([a-z])$', dir_name)
        if suffix_match:
            return f"{peptide}_{suffix_match.group(1)}"
        
        # If no suffix found, check if this is a duplicate by looking at the full path
        # This will be handled in the main analysis scripts
        return peptide
    return None

def classify_dna(dir_name):
    """
    Extract and standardize DNA name from directory name.
    Rules:
    1. Extract everything before the peptide name
    2. Remove "OPN", "DPN" (case-insensitive)
    3. Make everything case-insensitive
    4. Remove leading underscores
    5. Standardize hor7_40cen variants (including hor7_cen)
    6. Keep "DNA_" prefix if present to distinguish from non-DNA variants
    
    Args:
        dir_name (str): Directory name containing the DNA
    Returns:
        str or None: The standardized DNA name if found, None otherwise
    """
    # First find the peptide to know where to stop
    peptide = classify_peptide(dir_name)
    if not peptide:
        return None
        
    # Get everything before the peptide
    peptide_start = dir_name.find(peptide)
    if peptide_start == -1:
        return None
        
    dna_part = dir_name[:peptide_start].rstrip('_')
    
    # Remove OPN, DPN (case-insensitive) but keep DNA_ prefix
    dna_part = re.sub(r'(?i)^(OPN|DPN)_?', '', dna_part)
    
    # Remove leading underscores
    dna_part = dna_part.lstrip('_')
    
    # Convert to lowercase for standardization
    dna_part = dna_part.lower()
    
    # Standardize hor7_40cen variants
    if 'hor7' in dna_part:
        if 'cen' in dna_part:
            # Match hor7_40cen with optional number or hor7_cen
            if re.search(r'hor7_40cen(?:_\d+)?$', dna_part) or re.search(r'hor7_cen(?:_\d+)?$', dna_part):
                return 'hor7_40cen'
    
    return dna_part

# If run as a script, test both classifiers
if __name__ == "__main__":
    test_dir_names = [
        "DNA_40A_VPVPV_peptide_substracted",
        "DNA_40B_ELLSL_peptide_substracted",
        "40_1_ctrl_VPVPV_2mMNapho_10mMNaCl_pH6",
        "DPN_aim18_40S_VPVPV_2mMNapho_10mMNaCl_pH6",
        "OPN_hor7_40cen_VPVPV_2mMNapho_10mMNaCl_pH6",
        "OPN_hor7_40S_VPVPV_2mMNapho_10mMNaCl_pH6",
        "dna_40_2_vgvgv_2mmnapho_10mmnacl_ph6",
        "invalid_name",
        "DNA_40F_abcde_peptide_substracted",  # lowercase letters
        "DNA_40G_ABCD_peptide_substracted",   # 4 letters
        "DNA_40H_ABCDEF_peptide_substracted"  # 6 letters
    ]
    print("Testing peptide and DNA classification:")
    print("-" * 50)
    for dir_name in test_dir_names:
        peptide = classify_peptide(dir_name)
        dna = classify_dna(dir_name)
        print(f"Directory: {dir_name}")
        print(f"Found peptide: {peptide}")
        print(f"Found DNA: {dna}")
        print("-" * 50) 