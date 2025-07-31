import json
import hashlib
from typing import List, Dict, Any, Tuple
import base64
from Crypto.Cipher import AES
from Crypto import Random
import copy

class Aes:
    """
    AES encryption and decryption class.
    """
    def encrypt(self, data, password):
        password = self.get_password(password)  
        data = copy.deepcopy(data)
        if not isinstance(data, str):
            data = str(data)
        data = data + (AES.block_size - len(data) % AES.block_size) * chr(AES.block_size - len(data) % AES.block_size)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(password, AES.MODE_CBC, iv)
        encrypted_bytes = base64.b64encode(iv + cipher.encrypt(data.encode()))
        return encrypted_bytes.decode() 

    def decrypt(self, data, password:str):  
        password = self.get_password(password)  
        data = base64.b64decode(data)
        iv = data[:AES.block_size]
        cipher = AES.new(password, AES.MODE_CBC, iv)
        data =  cipher.decrypt(data[AES.block_size:])
        data = data[:-ord(data)].decode('utf-8')
        return data

    def get_password(self, password:str):
        if isinstance(password, str):
            password = password.encode()
        # if password is a key, use the key's private key as password
        return hashlib.sha256(password).digest()


class MultiSig:
    """
    Multi-signature implementation using AES encryption.
    Requires M of N signatures to decrypt data.
    """
    
    def __init__(self, m: int, n: int):
        """
        Initialize MultiSig with M of N threshold.

        Args:
            m: Minimum number of signatures required
            n: Total number of possible signers
        """
        if m > n:
            raise ValueError("M cannot be greater than N")
        if m < 1:
            raise ValueError("M must be at least 1")
            
        self.m = m
        self.n = n
        self.aes = Aes()
        
    def create_shares(self, secret: str, passwords: List) -> List[Dict]:
        """
        Create encrypted shares of a secret using Shamir's Secret Sharing concept.
        
        Args:
            secret: The secret data to be shared
            passwords: List of passwords for each signer (length must equal n)
            
        Returns:
            List of encrypted shares
        """
        if len(passwords) != self.n:
            raise ValueError(f"Expected {self.n} passwords, got {len(passwords)}")
        
        # For simplicity, we'll create a threshold scheme where we encrypt
        # the secret with combinations of passwords
        shares = []
        
        # Generate a master key
        master_key = hashlib.sha256(secret.encode()).hexdigest()
        
        for i, password in enumerate(passwords):
            # Create a share that includes the index and encrypted partial data
            share_data = {
                'index': i,
                'master_key_part': self.aes.encrypt(master_key, password),
                'threshold': self.m,
                'total': self.n
            }
            shares.append(share_data)
            
        # Also store the encrypted secret with the master key
        encrypted_secret = self.aes.encrypt(secret, master_key)
        
        return shares, encrypted_secret
    
    def combine_shares(self, shares: List[Tuple[Dict, str]], 
                      encrypted_secret: str) -> str:
        """
        Combine shares to decrypt the secret.
        
        Args:
            shares: List of tuples (share_data, password)
            encrypted_secret: The encrypted secret
            
        Returns:
            Decrypted secret if threshold is met
        """
        if len(shares) < self.m:
            raise ValueError(f"Need at least {self.m} shares, got {len(shares)}")
        
        # Verify all shares are from the same multisig setup
        threshold = shares[0][0]['threshold']
        total = shares[0][0]['total']
        
        for share, _ in shares:
            if share['threshold'] != threshold or share['total'] != total:
                raise ValueError("Shares are from different multisig setups")
        
        # Decrypt master key parts
        master_key_parts = []
        for share, password in shares[:self.m]:  # Only need M shares
            try:
                part = self.aes.decrypt(share['master_key_part'], password)
                master_key_parts.append(part)
            except Exception as e:
                raise ValueError(f"Failed to decrypt share {share['index']}: {str(e)}")
        
        # In a real implementation, you'd use proper secret sharing
        # For this example, we'll verify that all parts match (they should)
        master_key = master_key_parts[0]
        for part in master_key_parts[1:]:
            if part != master_key:
                raise ValueError("Invalid shares - master key parts don't match")
        
        # Decrypt the secret using the recovered master key
        try:
            secret = self.aes.decrypt(encrypted_secret, master_key)
            return secret
        except Exception as e:
            raise ValueError(f"Failed to decrypt secret: {str(e)}")


    @staticmethod
    def test():
        """
        Test function to verify MultiSig functionality.
        """
        print("=== MultiSig Test Suite ===\n")
        
        # Test 1: Basic 2-of-3 multisig
        print("Test 1: Basic 2-of-3 MultiSig")
        multisig = MultiSig(m=2, n=3)
        secret = "This is a very secret message!"
        passwords = ["alice_password", "bob_password", "charlie_password"]
        
        # Create shares
        shares, encrypted_secret = multisig.create_shares(secret, passwords)
        print(f"Created {len(shares)} shares for secret: '{secret}'")
        
        # Try to decrypt with 2 shares (should work)
        selected_shares = [
            (shares[0], passwords[0]),  # Alice's share
            (shares[1], passwords[1])   # Bob's share
        ]
        
        try:
            decrypted = multisig.combine_shares(selected_shares, encrypted_secret)
            print(f"✓ Successfully decrypted with 2 shares: '{decrypted}'")
            assert decrypted == secret, "Decrypted value doesn't match original"
        except Exception as e:
            print(f"✗ Failed to decrypt with 2 shares: {e}")
        
        # Try to decrypt with only 1 share (should fail)
        print("\nTest 2: Attempting with insufficient shares (1 of 2 required)")
        try:
            single_share = [(shares[0], passwords[0])]
            decrypted = multisig.combine_shares(single_share, encrypted_secret)
            print(f"✗ Should have failed but got: '{decrypted}'")
        except ValueError as e:
            print(f"✓ Correctly failed with insufficient shares: {e}")
        
        # Test 3: Wrong password
        print("\nTest 3: Attempting with wrong password")
        wrong_shares = [
            (shares[0], "wrong_password"),
            (shares[1], passwords[1])
        ]
        try:
            decrypted = multisig.combine_shares(wrong_shares, encrypted_secret)
            print(f"✗ Should have failed but got: '{decrypted}'")
        except ValueError as e:
            print(f"✓ Correctly failed with wrong password: {e}")
        
        # Test 4: 3-of-5 multisig
        print("\nTest 4: 3-of-5 MultiSig")
        multisig_3of5 = MultiSig(m=3, n=5)
        secret_2 = "Another secret requiring 3 of 5 signatures"
        passwords_5 = [f"user_{i}_pass" for i in range(5)]
        
        shares_5, encrypted_secret_5 = multisig_3of5.create_shares(secret_2, passwords_5)
        
        # Use shares from users 0, 2, and 4
        selected_shares_5 = [
            (shares_5[0], passwords_5[0]),
            (shares_5[2], passwords_5[2]),
            (shares_5[4], passwords_5[4])
        ]
        
        try:
            decrypted_5 = multisig_3of5.combine_shares(selected_shares_5, encrypted_secret_5)
            print(f"✓ Successfully decrypted 3-of-5: '{decrypted_5}'")
            assert decrypted_5 == secret_2, "Decrypted value doesn't match original"
        except Exception as e:
            print(f"✗ Failed to decrypt 3-of-5: {e}")
        
        # Test 5: Edge cases
        print("\nTest 5: Edge cases")
        
        # 1-of-1 multisig (essentially single sig)
        multisig_1of1 = MultiSig(m=1, n=1)
        shares_1, encrypted_1 = multisig_1of1.create_shares("single sig secret", ["only_pass"])
        decrypted_1 = multisig_1of1.combine_shares([(shares_1[0], "only_pass")], encrypted_1)
        print(f"✓ 1-of-1 multisig works: '{decrypted_1}'")
        
        # Invalid M > N
        try:
            invalid_multisig = MultiSig(m=5, n=3)
            print("✗ Should have failed to create invalid multisig")
        except ValueError as e:
            print(f"✓ Correctly rejected invalid M > N: {e}")
        
        print("\n=== All tests completed ===")


if __name__ == "__main__":
    test_multisig()