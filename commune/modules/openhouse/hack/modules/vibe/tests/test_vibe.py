import unittest
import random
from vibe import Vibe

class TestVibe(unittest.TestCase):
    
    def setUp(self):
        # Use a fixed seed for reproducible tests
        self.vibe = Vibe(seed=42)
    
    def test_initialization(self):
        """Test that the Vibe class initializes correctly"""
        self.assertIsNone(self.vibe.current_mood)
        self.assertIsNone(self.vibe.current_colors)
        self.assertIsNone(self.vibe.current_quote)
    
    def test_generate_vibe_random(self):
        """Test generating a random vibe"""
        vibe_result = self.vibe.generate_vibe()
        
        # Check that all expected keys are present
        self.assertIn('mood', vibe_result)
        self.assertIn('colors', vibe_result)
        self.assertIn('quote', vibe_result)
        self.assertIn('timestamp', vibe_result)
        
        # Check that the current state is updated
        self.assertEqual(self.vibe.current_mood, vibe_result['mood'])
        self.assertEqual(self.vibe.current_colors, vibe_result['colors'])
        self.assertEqual(self.vibe.current_quote, vibe_result['quote'])
    
    def test_generate_vibe_specific(self):
        """Test generating a specific vibe"""
        vibe_result = self.vibe.generate_vibe("cosmic")
        
        # Check that the mood is set correctly
        self.assertEqual(vibe_result['mood'], "cosmic")
        self.assertEqual(self.vibe.current_mood, "cosmic")
        
        # Check that colors match the cosmic palette
        self.assertEqual(vibe_result['colors'], self.vibe.COLOR_SCHEMES["cosmic"])
    
    def test_get_all_moods(self):
        """Test retrieving all available moods"""
        moods = self.vibe.get_all_moods()
        
        # Check that we get the expected number of moods
        self.assertEqual(len(moods), 10)
        
        # Check that all moods are strings
        for mood in moods:
            self.assertIsInstance(mood, str)
    
    def test_get_color_scheme(self):
        """Test retrieving color schemes"""
        # Test valid mood
        colors = self.vibe.get_color_scheme("chill")
        self.assertEqual(colors, self.vibe.COLOR_SCHEMES["chill"])
        
        # Test invalid mood
        with self.assertRaises(ValueError):
            self.vibe.get_color_scheme("invalid_mood")
    
    def test_get_quote(self):
        """Test retrieving quotes"""
        # Test valid mood
        quote = self.vibe.get_quote("zen")
        self.assertIn(quote, self.vibe.QUOTES["zen"])
        
        # Test invalid mood
        with self.assertRaises(ValueError):
            self.vibe.get_quote("invalid_mood")
    
    def test_vibe_check_no_vibe(self):
        """Test vibe check with no active vibe"""
        # Reset the vibe state
        self.vibe.current_mood = None
        self.vibe.current_colors = None
        self.vibe.current_quote = None
        
        check_result = self.vibe.vibe_check()
        
        # Should indicate no active vibe
        self.assertEqual(check_result['status'], "No vibe currently active")
        self.assertEqual(check_result['suggestion'], "Generate a vibe first")
    
    def test_vibe_check_with_vibe(self):
        """Test vibe check with an active vibe"""
        # Generate a vibe first
        self.vibe.generate_vibe("dreamy")
        
        check_result = self.vibe.vibe_check()
        
        # Should have passed the check
        self.assertEqual(check_result['status'], "Vibe check passed")
        
        # Should include current vibe info
        self.assertEqual(check_result['current_vibe']['mood'], "dreamy")
        
        # Should include metrics
        self.assertIn('intensity', check_result)
        self.assertIn('resonance', check_result)
    
    def test_mix_vibes(self):
        """Test mixing two vibes"""
        mixed_vibe = self.vibe.mix_vibes("retro", "futuristic")
        
        # Check hybrid name format
        self.assertEqual(mixed_vibe['hybrid_mood'], "ret-fut")
        
        # Check parent moods
        self.assertEqual(mixed_vibe['parent_moods'], ["retro", "futuristic"])
        
        # Check that colors are mixed from both parents
        self.assertEqual(len(mixed_vibe['colors']), 4)
        self.assertEqual(mixed_vibe['colors'][:2], self.vibe.COLOR_SCHEMES["retro"][:2])
        self.assertEqual(mixed_vibe['colors'][2:], self.vibe.COLOR_SCHEMES["futuristic"][:2])
        
        # Test with invalid moods
        with self.assertRaises(ValueError):
            self.vibe.mix_vibes("retro", "invalid_mood")
    
    def test_string_representation(self):
        """Test the string representation of a vibe"""
        # With no active vibe
        self.vibe.current_mood = None
        self.assertEqual(str(self.vibe), "No vibe currently active. Generate one with generate_vibe()")
        
        # With an active vibe
        self.vibe.generate_vibe("electric")
        string_rep = str(self.vibe)
        
        self.assertIn("ELECTRIC", string_rep)
        self.assertIn("Colors:", string_rep)
        self.assertIn("Quote:", string_rep)

if __name__ == '__main__':
    unittest.main()
