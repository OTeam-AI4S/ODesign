#!/usr/bin/env python
"""
Antibody Chain Identification and CDR Masking Tool

This script analyzes a PDB or CIF file to:
1. Identify each chain as antigen, heavy chain, or light chain
2. Extract sequences from antibody chains
3. Identify CDR regions in antibody chains
4. Generate masked sequences with CDRs replaced by "-"
5. Output results in JSON format compatible with ODesign

Usage:
    python identify_cdr.py <pdb_or_cif_file> [--scheme imgt] [--output output.json]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from abnumber import Chain
except ImportError:
    print("Error: 'abnumber' library not found.")
    print("Please install it using: pip install abnumber")
    sys.exit(1)

try:
    from Bio import PDB
    from Bio.PDB import PDBIO, MMCIFParser
except ImportError:
    print("Error: 'biopython' library not found.")
    print("Please install it using: pip install biopython")
    sys.exit(1)


class AntibodyChainAnalyzer:
    """Analyzes protein chains to identify antibody chains and their CDR regions."""

    CHAIN_TYPES = {
        'heavy': 'Heavy Chain',
        'light': 'Light Chain',
        'antigen': 'Antigen'
    }

    def __init__(self, scheme: str = 'imgt'):
        """
        Initialize the analyzer.

        Args:
            scheme: Numbering scheme for CDR identification (imgt, kabat, chothia, etc.)
        """
        self.scheme = scheme
        # self.parser will be initialized dynamically based on file type in analyze_pdb

    def extract_sequence_from_chain(self, chain) -> str:
        """
        Extract amino acid sequence from a PDB chain.

        Args:
            chain: Bio.PDB.Chain object

        Returns:
            str: Single-letter amino acid sequence
        """
        aa_dict = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }

        sequence = []
        for residue in chain:
            # Skip hetero atoms and water
            if residue.id[0] == ' ':
                resname = residue.get_resname()
                if resname in aa_dict:
                    sequence.append(aa_dict[resname])

        return ''.join(sequence)

    def identify_chain_type(self, sequence: str) -> Tuple[Optional[str], Optional[Chain]]:
        """
        Identify if a sequence is a heavy chain, light chain, or antigen.

        Args:
            sequence: Amino acid sequence

        Returns:
            Tuple of (chain_type, abnumber_chain)
            chain_type: 'heavy', 'light', or 'antigen'
            abnumber_chain: Chain object if antibody, None otherwise
        """
        if len(sequence) < 20:
            return 'antigen', None

        try:
            # Try to number as antibody sequence
            ab_chain = Chain(sequence, scheme=self.scheme)

            # Check chain type from abnumber
            if ab_chain.chain_type == 'H':
                return 'heavy', ab_chain
            elif ab_chain.chain_type == 'L' or ab_chain.chain_type == 'K':
                return 'light', ab_chain
            else:
                # If numbering succeeds but type is unknown, still might be antibody
                # Check for typical antibody length
                if 90 <= len(sequence) <= 130:
                    # Try to infer from CDR presence
                    if ab_chain.cdr1_seq or ab_chain.cdr2_seq or ab_chain.cdr3_seq:
                        # Guess heavy vs light based on length (rough heuristic)
                        if len(sequence) > 110:
                            return 'heavy', ab_chain
                        else:
                            return 'light', ab_chain

                return 'antigen', None

        except Exception:
            # If abnumber fails, it's likely an antigen
            return 'antigen', None

    def get_cdr_info(self, sequence: str, ab_chain: Chain) -> Dict:
        """
        Extract CDR information from an antibody chain.

        Args:
            sequence: Original amino acid sequence
            ab_chain: abnumber Chain object

        Returns:
            dict: CDR information including positions and sequences
        """
        cdrs = {}

        # Get the aligned sequence from abnumber
        # ab_chain.positions gives us the numbering for each position
        seq_positions = list(ab_chain.positions)
        aligned_seq = str(ab_chain.seq)

        # Define CDR regions based on scheme
        # IMGT definition (most common)
        cdr_definitions = {
            'imgt': {
                'CDR1': (27, 38),
                'CDR2': (56, 65),
                'CDR3': (105, 117)
            },
            'kabat': {
                'CDR1-H': (31, 35),  # Heavy chain
                'CDR2-H': (50, 65),
                'CDR3-H': (95, 102),
                'CDR1-L': (24, 34),  # Light chain
                'CDR2-L': (50, 56),
                'CDR3-L': (89, 97)
            }
        }

        # Use abnumber's built-in CDR detection
        if ab_chain.cdr1_seq:
            start_idx = sequence.find(ab_chain.cdr1_seq)
            if start_idx != -1:
                cdrs['CDR1'] = {
                    'sequence': ab_chain.cdr1_seq,
                    'start': start_idx,
                    'end': start_idx + len(ab_chain.cdr1_seq),
                    'length': len(ab_chain.cdr1_seq)
                }

        if ab_chain.cdr2_seq:
            start_idx = sequence.find(ab_chain.cdr2_seq)
            if start_idx != -1:
                cdrs['CDR2'] = {
                    'sequence': ab_chain.cdr2_seq,
                    'start': start_idx,
                    'end': start_idx + len(ab_chain.cdr2_seq),
                    'length': len(ab_chain.cdr2_seq)
                }

        if ab_chain.cdr3_seq:
            start_idx = sequence.find(ab_chain.cdr3_seq)
            if start_idx != -1:
                cdrs['CDR3'] = {
                    'sequence': ab_chain.cdr3_seq,
                    'start': start_idx,
                    'end': start_idx + len(ab_chain.cdr3_seq),
                    'length': len(ab_chain.cdr3_seq)
                }

        return cdrs

    def mask_cdr_regions(self, sequence: str, cdr_info: Dict) -> str:
        """
        Replace CDR regions with "-" in the sequence.

        Args:
            sequence: Original amino acid sequence
            cdr_info: Dictionary containing CDR positions

        Returns:
            str: Sequence with CDRs masked by "-"
        """
        # Convert to list for easier manipulation
        masked_seq = list(sequence)

        # Mask each CDR region
        for cdr_name in sorted(cdr_info.keys()):
            cdr = cdr_info[cdr_name]
            start = cdr['start']
            end = cdr['end']
            # Replace with a single "-"
            for i in range(start, end):
                if i < len(masked_seq):
                    masked_seq[i] = ''
            # Insert single "-" at the start position
            if start < len(masked_seq):
                masked_seq[start] = '-'

        # Remove empty strings and join
        masked_seq = ''.join(masked_seq)
        return masked_seq

    def get_cdr_length_ranges(self, cdr_info: Dict) -> str:
        """
        Generate length range string for CDRs (e.g., "6-7,6-7,9-15").

        Args:
            cdr_info: Dictionary containing CDR information

        Returns:
            str: Comma-separated length ranges
        """
        lengths = []
        for cdr_name in ['CDR1', 'CDR2', 'CDR3']:
            if cdr_name in cdr_info:
                length = cdr_info[cdr_name]['length']
                # Define reasonable range around the actual length
                min_len = max(3, length - 2)
                max_len = length + 3
                lengths.append(f"{min_len}-{max_len}")
            else:
                # Default ranges if CDR not found
                if cdr_name == 'CDR3':
                    lengths.append("9-15")
                else:
                    lengths.append("6-7")

        return ','.join(lengths)

    def analyze_pdb(self, pdb_path: str) -> Dict:
        """
        Analyze all chains in a PDB or CIF file.

        Args:
            pdb_path: Path to PDB or CIF file

        Returns:
            dict: Analysis results for all chains
        """
        if str(pdb_path).lower().endswith('.cif'):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDB.PDBParser(QUIET=True)
            
        structure = parser.get_structure('protein', pdb_path)
        results = {
            'pdb_file': pdb_path,
            'chains': {}
        }

        # Iterate through all chains in the structure
        for model in structure:
            for chain in model:
                chain_id = chain.id
                sequence = self.extract_sequence_from_chain(chain)

                if not sequence:
                    continue

                # Identify chain type
                chain_type, ab_chain = self.identify_chain_type(sequence)

                chain_info = {
                    'chain_id': chain_id,
                    'chain_type': self.CHAIN_TYPES[chain_type],
                    'sequence': sequence,
                    'length': len(sequence)
                }

                # If it's an antibody chain, get CDR info
                if chain_type in ['heavy', 'light'] and ab_chain:
                    cdr_info = self.get_cdr_info(sequence, ab_chain)
                    masked_sequence = self.mask_cdr_regions(sequence, cdr_info)
                    length_ranges = self.get_cdr_length_ranges(cdr_info)

                    chain_info.update({
                        'antibody_type': chain_type,
                        'cdrs': cdr_info,
                        'masked_sequence': masked_sequence,
                        'cdr_length_ranges': length_ranges,
                        'num_cdrs': len(cdr_info)
                    })

                results['chains'][chain_id] = chain_info

        return results

    def generate_odesign_json(self, analysis_results: Dict,
                             antigen_pdb: str,
                             hotspot: str = "",
                             name: str = "antibody_design") -> List[Dict]:
        """
        Generate ODesign-compatible JSON configuration.

        Args:
            analysis_results: Results from analyze_pdb
            antigen_pdb: Path to antigen PDB file
            hotspot: Hotspot residues (e.g., "A/538,A/151")
            name: Name for the design task

        Returns:
            list: ODesign configuration
        """
        chains_config = []

        # Add antibody chains
        for chain_id, chain_info in analysis_results['chains'].items():
            if 'antibody_type' in chain_info:
                chains_config.append({
                    'chain_type': 'proteinChain',
                    'im': 'antibody',
                    'sequence': chain_info['masked_sequence'],
                    'length': chain_info['cdr_length_ranges']
                })

        # Add antigen chain
        antigen_chains = [c for c in analysis_results['chains'].values()
                         if c['chain_type'] == 'Antigen']

        if antigen_chains:
            # Use first antigen chain
            antigen_chain = antigen_chains[0]
            antigen_seq = f"{antigen_chain['chain_id']}/1-{antigen_chain['length']}"

            chains_config.append({
                'chain_type': 'proteinChain',
                'im': 'antigen',
                'sequence': antigen_seq
            })

        config = [{
            'name': name,
            'antigen': antigen_pdb,
            'hotspot': hotspot,
            'chains': chains_config
        }]

        return config


def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description='Identify chain types and CDR regions in a PDB or CIF file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('pdb_file', type=str, help='Path to PDB or CIF file')
    parser.add_argument('--scheme', type=str, default='imgt',
                       choices=['imgt', 'kabat', 'chothia', 'aho'],
                       help='Numbering scheme for CDR identification (default: imgt)')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--odesign', action='store_true',
                       help='Generate ODesign-compatible JSON output')
    parser.add_argument('--hotspot', type=str, default='',
                       help='Hotspot residues for ODesign (e.g., "A/538,A/151")')
    parser.add_argument('--name', type=str, default='antibody_design',
                       help='Name for the design task')

    args = parser.parse_args()

    # Check if PDB file exists
    if not Path(args.pdb_file).exists():
        print(f"Error: Structure file not found: {args.pdb_file}")
        sys.exit(1)

    # Analyze the PDB file
    analyzer = AntibodyChainAnalyzer(scheme=args.scheme)

    print(f"Analyzing structure file: {args.pdb_file}")
    print(f"Using numbering scheme: {args.scheme}")
    print("-" * 60)

    results = analyzer.analyze_pdb(args.pdb_file)

    # Print results
    print(f"\nFound {len(results['chains'])} chain(s):\n")

    for chain_id, chain_info in results['chains'].items():
        print(f"Chain {chain_id}:")
        print(f"  Type: {chain_info['chain_type']}")
        print(f"  Length: {chain_info['length']} residues")
        print(f"  Sequence: {chain_info['sequence'][:50]}{'...' if len(chain_info['sequence']) > 50 else ''}")

        if 'antibody_type' in chain_info:
            print(f"  Antibody Type: {chain_info['antibody_type']}")
            print(f"  CDRs Found: {chain_info['num_cdrs']}")

            for cdr_name, cdr_data in chain_info['cdrs'].items():
                print(f"    {cdr_name}: {cdr_data['sequence']} "
                      f"(pos {cdr_data['start']+1}-{cdr_data['end']})")

            print(f"  Masked Sequence: {chain_info['masked_sequence']}")
            print(f"  CDR Length Ranges: {chain_info['cdr_length_ranges']}")

        print()

    # Generate output
    if args.odesign:
        output_data = analyzer.generate_odesign_json(
            results,
            args.pdb_file,
            args.hotspot,
            args.name
        )
    else:
        output_data = results

    # Save to file if specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Results saved to: {output_path}")
    else:
        # Print JSON to stdout
        print("\nJSON Output:")
        print("-" * 60)
        print(json.dumps(output_data, indent=4))


if __name__ == '__main__':
    main()
