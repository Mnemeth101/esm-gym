import unittest
from unittest.mock import MagicMock, patch
from esm.sdk.api import ESM3InferenceClient, ESMProtein, LogitsConfig, GenerationConfig, ESMProteinError
import os
import json
import tempfile
import torch
import numpy as np
from typing import List, Dict, Any, Optional

from benchmarking import (
    single_metric_UACCE,
    single_metric_average_pLDDT,
    single_metric_pTM,
    single_metric_foldability,
    aggregated_metric_entropy,
    aggregated_metric_diversity,
    aggregated_cosine_similarities,
    BenchmarkRunner,
    load_protein_from_fasta,
)

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

class TestBenchmarkingUtils(unittest.TestCase):

    def setUp(self):
        # Mock ESM3InferenceClient and ESMProtein
        self.mock_client = MagicMock(spec=ESM3InferenceClient)
        self.mock_protein = MagicMock(spec=ESMProtein)
        
        # Sample protein sequences for aggregated metrics tests
        self.protein_sequences = [
            "ACDEFGHIKL",
            "ACDEFGHIKM",
            "ACDEFGHIKP",
            "ACDEFGHIKS"
        ]

    def test_single_metric_UACCE(self):
        # Mock client.encode and logits
        mock_tensor = MagicMock()
        mock_tensor.sequence.numpy.return_value = [1, 2, 3, 4]
        self.mock_client.encode.return_value = mock_tensor

        mock_logits = MagicMock()
        mock_logits.sequence = MagicMock()
        mock_logits.sequence.shape = (4, 20)
        self.mock_client.logits.return_value = MagicMock(logits=mock_logits)

        # Mock softmax probabilities
        import torch.nn.functional as F
        mock_logits.sequence.cpu.return_value.numpy.return_value = [[0.1] * 20 for _ in range(4)]
        
        # Set a length for the mock protein to avoid division by zero
        self.mock_protein.__len__.return_value = 4
        
        # Call the function
        result = single_metric_UACCE(self.mock_client, self.mock_protein, verbose=False)

        # Assert the result is a float
        self.assertIsInstance(result, float)

    def test_single_metric_average_pLDDT(self):
        # Mock pLDDT values
        self.mock_protein.plddt = MagicMock()
        self.mock_protein.plddt.mean.return_value = 85.0

        # Call the function
        result = single_metric_average_pLDDT(self.mock_protein)

        # Assert the result is correct
        self.assertEqual(result, 85.0)

    def test_single_metric_average_pLDDT_no_plddt(self):
        # Mock pLDDT as None
        self.mock_protein.plddt = None

        # Assert ValueError is raised
        with self.assertRaises(ValueError):
            single_metric_average_pLDDT(self.mock_protein)

    def test_single_metric_pTM(self):
        # Mock pTM value
        self.mock_protein.ptm = 0.75

        # Call the function
        result = single_metric_pTM(self.mock_protein)

        # Assert the result is correct
        self.assertEqual(result, 0.75)

    def test_single_metric_pTM_no_ptm(self):
        # Mock pTM as None
        self.mock_protein.ptm = None

        # Assert ValueError is raised
        with self.assertRaises(ValueError):
            single_metric_pTM(self.mock_protein)

    def test_single_metric_foldability(self):
        # Mock protein sequence
        self.mock_protein.sequence = "ACDEFGHIKLMNPQRSTVWY"

        # Mock client.generate
        mock_generated_protein = MagicMock(spec=ESMProtein)
        mock_generated_protein.plddt.mean.return_value = 85.0
        mock_generated_protein.ptm = 0.8
        self.mock_client.generate.return_value = mock_generated_protein

        # Call the function
        result = single_metric_foldability(self.mock_client, self.mock_protein, num_samples=3, verbose=False)

        # Assert the result is a float and within valid range
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_single_metric_foldability_no_sequence(self):
        # Mock protein with no sequence
        self.mock_protein.sequence = None

        # Assert ValueError is raised
        with self.assertRaises(ValueError):
            single_metric_foldability(self.mock_client, self.mock_protein, num_samples=3, verbose=False)

    def test_single_metric_foldability_perfect_score(self):
        """Test foldability with all samples meeting criteria (100% score)"""
        # Mock protein sequence
        self.mock_protein.sequence = "ACDEFGHIKLMNPQRSTVWY"
        self.mock_protein.copy.return_value = self.mock_protein

        # Mock client.generate with good scores
        mock_generated_protein = MagicMock(spec=ESMProtein)
        mock_generated_protein.plddt = MagicMock()
        mock_generated_protein.plddt.mean.return_value = 90.0  # > 80 threshold
        mock_generated_protein.ptm = 0.85  # > 0.7 threshold
        self.mock_client.generate.return_value = mock_generated_protein

        # Call the function
        result = single_metric_foldability(self.mock_client, self.mock_protein, num_samples=5, verbose=False)

        # All samples should pass, resulting in a foldability score of 1.0
        self.assertEqual(result, 1.0)
        # Check that generate was called the right number of times
        self.assertEqual(self.mock_client.generate.call_count, 5)

    def test_single_metric_foldability_zero_score(self):
        """Test foldability with no samples meeting criteria (0% score)"""
        # Mock protein sequence
        self.mock_protein.sequence = "ACDEFGHIKLMNPQRSTVWY"
        self.mock_protein.copy.return_value = self.mock_protein

        # Mock client.generate with poor scores
        mock_generated_protein = MagicMock(spec=ESMProtein)
        mock_generated_protein.plddt = MagicMock()
        mock_generated_protein.plddt.mean.return_value = 75.0  # < 80 threshold
        mock_generated_protein.ptm = 0.65  # < 0.7 threshold
        self.mock_client.generate.return_value = mock_generated_protein

        # Call the function
        result = single_metric_foldability(self.mock_client, self.mock_protein, num_samples=5, verbose=False)

        # No samples should pass, resulting in a foldability score of 0.0
        self.assertEqual(result, 0.0)

    def test_single_metric_foldability_mixed_results(self):
        """Test foldability with some samples meeting criteria and some failing"""
        # Mock protein sequence
        self.mock_protein.sequence = "ACDEFGHIKLMNPQRSTVWY"
        self.mock_protein.copy.return_value = self.mock_protein

        # Create two different result proteins - one good, one bad
        good_protein = MagicMock(spec=ESMProtein)
        good_protein.plddt = MagicMock()
        good_protein.plddt.mean.return_value = 85.0  # > 80 threshold
        good_protein.ptm = 0.75  # > 0.7 threshold

        bad_protein = MagicMock(spec=ESMProtein)
        bad_protein.plddt = MagicMock()
        bad_protein.plddt.mean.return_value = 70.0  # < 80 threshold
        bad_protein.ptm = 0.65  # < 0.7 threshold

        # Mock generate to return alternating results
        self.mock_client.generate.side_effect = [good_protein, bad_protein, good_protein, bad_protein]

        # Call the function with 4 samples
        result = single_metric_foldability(self.mock_client, self.mock_protein, num_samples=4, verbose=False)

        # Should have 2 passing samples out of 4, resulting in a foldability score of 0.5
        self.assertEqual(result, 0.5)

    def test_single_metric_foldability_generation_error(self):
        """Test handling of errors during structure generation"""
        # Mock protein sequence
        self.mock_protein.sequence = "ACDEFGHIKLMNPQRSTVWY"
        self.mock_protein.copy.return_value = self.mock_protein
        
        # Create a good protein for the first call
        good_protein = MagicMock(spec=ESMProtein)
        good_protein.plddt = MagicMock()
        good_protein.plddt.mean.return_value = 85.0
        good_protein.ptm = 0.8
        
        # For testing ESMProteinError handling, we'll patch the isinstance function
        # in the specific module where it's used to control its behavior
        with patch('benchmarking.metrics.isinstance') as mock_isinstance:
            # Set up the mock to return True only when checking if something is an ESMProteinError
            def side_effect(obj, class_type):
                if class_type == ESMProteinError:
                    # Second call to generate returns an object that should be treated as ESMProteinError
                    return self.mock_client.generate.call_count == 2
                # Otherwise use the real isinstance
                return isinstance(obj, class_type)
                
            mock_isinstance.side_effect = side_effect
            
            # Make the first generate call return a good protein, second one can be anything
            self.mock_client.generate.side_effect = [good_protein, MagicMock()]
            
            # Call the function with 2 samples
            result = single_metric_foldability(self.mock_client, self.mock_protein, num_samples=2, verbose=False)
            
            # Check that we got 0.5 (1 success out of 2)
            self.assertEqual(result, 0.5)

    def test_single_metric_foldability_with_verbose(self):
        """Test foldability with verbose flag enabled"""
        # Mock protein sequence
        self.mock_protein.sequence = "ACDEFGHIKLMNPQRSTVWY"
        self.mock_protein.copy.return_value = self.mock_protein

        # Mock client.generate
        mock_generated_protein = MagicMock(spec=ESMProtein)
        mock_generated_protein.plddt = MagicMock()
        mock_generated_protein.plddt.mean.return_value = 85.0
        mock_generated_protein.ptm = 0.8
        self.mock_client.generate.return_value = mock_generated_protein

        # Call the function with verbose=True (coverage test)
        result = single_metric_foldability(self.mock_client, self.mock_protein, num_samples=1, verbose=True)

        # Should still return a valid result
        self.assertIsInstance(result, float)
        self.assertEqual(result, 1.0)

    def test_aggregated_metric_entropy_position_specific(self):
        """Test position-specific entropy calculation"""
        # Call the function
        result = aggregated_metric_entropy(self.protein_sequences, position_specific=True, verbose=False)
        
        # Check the results
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 10)  # Should have one entropy value per position
        
        # The last position should have high entropy (4 different amino acids)
        self.assertGreater(result[-1], 1.5)
        
        # First positions should have zero entropy (same amino acid in all sequences)
        self.assertEqual(result[0], 0.0)
        
    def test_aggregated_metric_entropy_global(self):
        """Test global entropy calculation"""
        # Call the function
        result = aggregated_metric_entropy(self.protein_sequences, position_specific=False, verbose=False)
        
        # Check the results
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        
    def test_aggregated_metric_entropy_verbose(self):
        """Test entropy calculation with verbose output"""
        # Call the function with verbose=True (coverage test)
        result_position = aggregated_metric_entropy(self.protein_sequences, position_specific=True, verbose=True)
        result_global = aggregated_metric_entropy(self.protein_sequences, position_specific=False, verbose=True)
        
        # Should still return valid results
        self.assertIsInstance(result_position, list)
        self.assertIsInstance(result_global, float)
        
    def test_aggregated_metric_entropy_empty_input(self):
        """Test entropy calculation with empty input"""
        # Should raise ValueError
        with self.assertRaises(ValueError):
            aggregated_metric_entropy([], position_specific=True)
            
    def test_aggregated_metric_entropy_unequal_lengths(self):
        """Test position-specific entropy with sequences of different lengths"""
        unequal_sequences = ["ACDEFG", "ACDEFGH", "ACDEFGHI"]
        
        # Should raise ValueError for position-specific entropy
        with self.assertRaises(ValueError):
            aggregated_metric_entropy(unequal_sequences, position_specific=True)
            
        # Should still work for global entropy
        result = aggregated_metric_entropy(unequal_sequences, position_specific=False)
        self.assertIsInstance(result, float)
            
    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation"""
        # Test with identical sequences
        self.assertEqual(levenshtein_distance("ABCDEFG", "ABCDEFG"), 0)
        
        # Test with simple substitution
        self.assertEqual(levenshtein_distance("ABCDEFG", "ABCXEFG"), 1)
        
        # Test with insertion
        self.assertEqual(levenshtein_distance("ABCDEFG", "ABCDEFGX"), 1)
        
        # Test with deletion
        self.assertEqual(levenshtein_distance("ABCDEFG", "ABCDEF"), 1)
        
        # Test with multiple operations
        self.assertEqual(levenshtein_distance("KITTEN", "SITTING"), 3)
        
    def test_aggregated_metric_diversity_levenshtein(self):
        """Test diversity calculation using Levenshtein distance"""
        # Call the function
        result = aggregated_metric_diversity(self.protein_sequences, method="levenshtein", verbose=False)
        
        # Check the results
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)  # Some diversity should exist
        self.assertLess(result, 0.5)     # Sequences are similar, so diversity should be low
        
    def test_aggregated_metric_diversity_hamming(self):
        """Test diversity calculation using Hamming distance"""
        # Call the function
        result = aggregated_metric_diversity(self.protein_sequences, method="hamming", verbose=False)
        
        # Check the results
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)  # Some diversity should exist
        self.assertLess(result, 0.5)     # Only last character differs
        
    def test_aggregated_metric_diversity_identity(self):
        """Test diversity calculation using identity percentage"""
        # Call the function
        result = aggregated_metric_diversity(self.protein_sequences, method="identity", verbose=False)
        
        # Check the results
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)  # Some diversity should exist
        self.assertLess(result, 0.5)     # Sequences are similar
        
    def test_aggregated_metric_diversity_verbose(self):
        """Test diversity calculation with verbose output"""
        # Call the function with verbose=True (coverage test)
        result = aggregated_metric_diversity(self.protein_sequences, method="levenshtein", verbose=True)
        
        # Should still return a valid result
        self.assertIsInstance(result, float)
        
    def test_aggregated_metric_diversity_invalid_method(self):
        """Test diversity calculation with invalid method"""
        # Should raise ValueError
        with self.assertRaises(ValueError):
            aggregated_metric_diversity(self.protein_sequences, method="invalid_method")
            
    def test_aggregated_metric_diversity_insufficient_sequences(self):
        """Test diversity calculation with insufficient sequences"""
        # Should raise ValueError
        with self.assertRaises(ValueError):
            aggregated_metric_diversity(["ACDEFGHIKL"], method="levenshtein")
            
    def test_aggregated_metric_diversity_unequal_lengths_hamming(self):
        """Test diversity with sequences of different lengths using Hamming distance"""
        unequal_sequences = ["ACDEFG", "ACDEFGH"]
        
        # Should raise ValueError for Hamming distance
        with self.assertRaises(ValueError):
            aggregated_metric_diversity(unequal_sequences, method="hamming")
            
    def test_aggregated_metric_diversity_unequal_lengths_identity(self):
        """Test diversity with sequences of different lengths using identity percentage"""
        unequal_sequences = ["ACDEFG", "ACDEFGH"]
        
        # Should raise ValueError for identity calculation
        with self.assertRaises(ValueError):
            aggregated_metric_diversity(unequal_sequences, method="identity")

    def test_aggregated_cosine_similarities(self):
        """Test cosine similarity calculation between protein embeddings"""
        # Mock forge client and its responses
        mock_forge_client = MagicMock()
        mock_forge_client.encode.return_value = MagicMock()
        mock_forge_client.logits.return_value = MagicMock(
            embeddings=torch.randn(1, 1, 2560)  # Mock embedding tensor
        )
        
        # Mock environment variable
        with patch.dict('os.environ', {'ESM_FORGE_API_KEY': 'test_token'}):
            with patch('benchmarking.metrics.ESM3ForgeInferenceClient', return_value=mock_forge_client):
                # Test with sample sequences
                original_protein = "ACDEFGHIKL"
                proteins = ["ACDEFGHIKM", "ACDEFGHIKP", "ACDEFGHIKS"]
                
                # Call the function
                similarities = aggregated_cosine_similarities(proteins, original_protein, verbose=False)
                
                # Check results
                self.assertIsInstance(similarities, list)
                self.assertEqual(len(similarities), len(proteins))
                self.assertTrue(all(isinstance(s, float) for s in similarities))
                eps = 1e-6
                self.assertTrue(all(-1.0 - eps <= s <= 1.0 + eps) for s in similarities) # Cosine similarity range

    def test_load_protein_from_fasta(self):
        """Test loading protein from FASTA file"""
        # Create a temporary FASTA file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">test_protein\nACDEFGHIKL\n")
            fasta_path = f.name
        
        try:
            # Load protein
            protein = load_protein_from_fasta(fasta_path)
            
            # Check results
            self.assertIsInstance(protein, ESMProtein)
            self.assertEqual(protein.sequence, "ACDEFGHIKL")
        finally:
            # Clean up
            os.unlink(fasta_path)

if __name__ == "__main__":
    unittest.main()