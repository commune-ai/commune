import numpy as np
from typing import Dict, Any

class VestingCurves:
    """Mathematical implementation of TGE vesting and multiplier curves"""
    
    @staticmethod
    def lock_multiplier(lock_days: int) -> float:
        """Calculate lock multiplier (0x -> 4x quadratic curve)
        
        Args:
            lock_days: Number of days tokens are locked
            
        Returns:
            Multiplier value (0 to 4)
        """
        if lock_days <= 0:
            return 0.0
        years = lock_days / 365
        return years ** 2
    
    @staticmethod
    def vested_tokens(days_elapsed: int, lock_period_days: int, base_tokens: float) -> float:
        """Calculate linearly vested tokens
        
        Args:
            days_elapsed: Days since lock started
            lock_period_days: Total lock period
            base_tokens: Initial token allocation
            
        Returns:
            Number of vested tokens
        """
        if lock_period_days <= 0:
            return 0.0
        
        multiplier = VestingCurves.lock_multiplier(lock_period_days)
        total_tokens = base_tokens * multiplier
        
        progress = min(days_elapsed / lock_period_days, 1.0)
        return total_tokens * progress
    
    @staticmethod
    def phase2_distribution(initial_allocation: float, lock_days: int) -> Dict[str, Any]:
        """Calculate full Phase 2 distribution details
        
        Args:
            initial_allocation: Base token amount
            lock_days: Lock period in days
            
        Returns:
            Dictionary with distribution details
        """
        multiplier = VestingCurves.lock_multiplier(lock_days)
        total_tokens = initial_allocation * multiplier
        
        return {
            'base_allocation': initial_allocation,
            'lock_days': lock_days,
            'lock_years': lock_days / 365,
            'multiplier': multiplier,
            'total_tokens': total_tokens,
            'vesting_type': 'linear',
            'daily_unlock': total_tokens / lock_days if lock_days > 0 else 0
        }
    
    @staticmethod
    def phase3_mint_amount(usd_value: float, mod_price: float) -> Dict[str, Any]:
        """Calculate Phase 3 minting with dynamic pricing
        
        Args:
            usd_value: USD contributed to protocol
            mod_price: Current MOD token price
            
        Returns:
            Minting details with adjustments
        """
        base_mod = usd_value  # 1 USD = 1 MOD base
        
        if mod_price < 1.0:
            adjustment = 1.10  # 10% surcharge
            reason = 'Below peg - buy from DEX encouraged'
        elif mod_price > 1.0:
            adjustment = 0.90  # 10% discount
            reason = 'Above peg - minting encouraged'
        else:
            adjustment = 1.0
            reason = 'At peg'
        
        mod_minted = base_mod * adjustment
        
        return {
            'usd_contributed': usd_value,
            'mod_price': mod_price,
            'base_mod': base_mod,
            'adjustment_factor': adjustment,
            'mod_minted': mod_minted,
            'reason': reason
        }
    
    @staticmethod
    def daily_mint_cap(days_since_genesis: int, max_supply: int = 42_000_000) -> float:
        """Calculate daily mint cap using Bitcoin halving schedule
        
        Args:
            days_since_genesis: Days since protocol launch
            max_supply: Maximum token supply
            
        Returns:
            Daily mint cap
        """
        halving_period = 1460  # ~4 years in days
        halvings = days_since_genesis // halving_period
        
        base_daily_cap = 14400  # 2 * 7200 blocks
        current_cap = base_daily_cap / (2 ** halvings)
        
        return max(current_cap, 1)  # Minimum 1 token per day
