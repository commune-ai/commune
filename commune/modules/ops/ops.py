import pytest
import numpy as np
import random
import time

class Ops:
    """Test suite for ComputationVerifier"""
    
    @pytest.fixture
    def verifier(self):
        """Create a ComputationVerifier instance for testing"""
        return ComputationVerifier(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            similarity_threshold=0.9,
            default_verification_runs=3
        )
    
    def test_initialization(self, verifier):
        """Test that the verifier initializes correctly"""
        assert verifier.similarity_threshold == 0.9
        assert verifier.default_verification_runs == 3
        assert len(verifier.verification_history) == 0
    
    def test_compute_embedding(self, verifier):
        """Test computing embeddings"""
        embedding = verifier.compute_embedding("Test string")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] > 0
    
    def test_compare_embeddings(self, verifier):
        """Test comparing embeddings"""
        embedding1 = verifier.compute_embedding("Hello world")
        embedding2 = verifier.compute_embedding("Hello world")
        embedding3 = verifier.compute_embedding("Completely different text")
        
        # Same text should have high similarity
        similarity_same = verifier.compare_embeddings(embedding1, embedding2)
        assert similarity_same > 0.9
        
        # Different text should have lower similarity
        similarity_diff = verifier.compare_embeddings(embedding1, embedding3)
        assert similarity_diff < similarity_same
    
    def test_verify_consistent_function(self, verifier):
        """Test verifying a consistent function"""
        def consistent_function(x, y):
            return x + y
        
        verification = verifier.verify(
            function=consistent_function,
            args=[5, 10],
            verification_runs=3
        )
        
        assert verification["is_valid"] == True
        assert verification["confidence"] > 0.9
        assert verification["function"] == "consistent_function"
        assert verification["original_result"] == 15
    
    def test_verify_inconsistent_function(self, verifier):
        """Test verifying a function with inconsistent results"""
        def inconsistent_function(x, y):
            return x + y + random.randint(1, 100)
        
        # First run to get original result
        original_result = inconsistent_function(5, 10)
        
        verification = verifier.verify(
            function=inconsistent_function,
            args=[5, 10],
            result=original_result,
            verification_runs=3
        )
        
        # Results should be inconsistent due to random component
        assert verification["average_similarity"] < 1.0
    
    def test_verification_timeout(self, verifier):
        """Test that verification respects timeout"""
        def slow_function(x):
            time.sleep(1)
            return x * 2
        
        start_time = time.time()
        verification = verifier.verify(
            function=slow_function,
            args=[5],
            verification_runs=10,
            timeout=2
        )
        
        duration = time.time() - start_time
        # Should timeout after about 2 seconds
        assert duration < 3.5
        # Should complete fewer than the requested 10 runs
        assert verification["verification_runs"] < 10
    
    def test_verification_history(self, verifier):
        """Test that verification history is maintained"""
        def test_function(x):
            return x * 2
        
        # Run a few verifications
        for i in range(3):
            verifier.verify(function=test_function, args=[i])
        
        # Check history
        history = verifier.get_verification_history()
        assert len(history) == 3
        
        # Check stats
        stats = verifier.get_verification_stats()
        assert stats["total_verifications"] == 3
        assert "average_confidence" in stats
        
        # Test clearing history
        verifier.clear_verification_history()
        assert len(verifier.get_verification_history()) == 0
    
    def test_convert_to_string(self, verifier):
        """Test converting different data types to strings"""
        # Test simple types
        assert verifier._convert_to_string(123) == "123"
        assert verifier._convert_to_string(None) == "None"
        
        # Test complex types
        list_str = verifier._convert_to_string([1, 2, 3])
        assert "[" in list_str and "]" in list_str
        
        dict_str = verifier._convert_to_string({"a": 1, "b": 2})
        assert "{" in dict_str and "}" in dict_str