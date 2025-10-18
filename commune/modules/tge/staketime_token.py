import commune as c
from typing import Dict, Any, Optional

class StaketimeToken(c.Module):
    """Tokenized vesting positions for Phase 2 rewards"""
    
    def __init__(self):
        self.positions = {}  # position_id -> position_data
        self.next_position_id = 1
    
    def create_position(self, 
                       owner: str, 
                       base_tokens: float, 
                       lock_days: int,
                       start_timestamp: int) -> Dict[str, Any]:
        """Create a new vesting position
        
        Args:
            owner: Wallet address
            base_tokens: Initial token allocation
            lock_days: Lock period in days
            start_timestamp: Unix timestamp of lock start
            
        Returns:
            Position details with unique ID
        """
        from tge.curves import VestingCurves
        
        multiplier = VestingCurves.lock_multiplier(lock_days)
        total_tokens = base_tokens * multiplier
        
        position = {
            'id': self.next_position_id,
            'owner': owner,
            'base_tokens': base_tokens,
            'lock_days': lock_days,
            'multiplier': multiplier,
            'total_tokens': total_tokens,
            'start_timestamp': start_timestamp,
            'end_timestamp': start_timestamp + (lock_days * 86400),
            'claimed_tokens': 0,
            'status': 'active'
        }
        
        self.positions[self.next_position_id] = position
        self.next_position_id += 1
        
        return position
    
    def claim_vested(self, position_id: int, current_timestamp: int) -> Dict[str, Any]:
        """Claim vested tokens from a position
        
        Args:
            position_id: Position identifier
            current_timestamp: Current Unix timestamp
            
        Returns:
            Claim details
        """
        from tge.curves import VestingCurves
        
        position = self.positions.get(position_id)
        if not position:
            return {'success': False, 'error': 'Position not found'}
        
        days_elapsed = (current_timestamp - position['start_timestamp']) // 86400
        
        vested = VestingCurves.vested_tokens(
            days_elapsed=days_elapsed,
            lock_period_days=position['lock_days'],
            base_tokens=position['base_tokens']
        )
        
        claimable = vested - position['claimed_tokens']
        
        if claimable <= 0:
            return {'success': False, 'error': 'No tokens to claim'}
        
        position['claimed_tokens'] += claimable
        
        if position['claimed_tokens'] >= position['total_tokens']:
            position['status'] = 'completed'
        
        return {
            'success': True,
            'position_id': position_id,
            'claimed': claimable,
            'total_claimed': position['claimed_tokens'],
            'remaining': position['total_tokens'] - position['claimed_tokens'],
            'status': position['status']
        }
    
    def transfer_position(self, position_id: int, new_owner: str) -> Dict[str, Any]:
        """Transfer vesting position to new owner (tradeable)
        
        Args:
            position_id: Position identifier
            new_owner: New wallet address
            
        Returns:
            Transfer confirmation
        """
        position = self.positions.get(position_id)
        if not position:
            return {'success': False, 'error': 'Position not found'}
        
        old_owner = position['owner']
        position['owner'] = new_owner
        
        return {
            'success': True,
            'position_id': position_id,
            'from': old_owner,
            'to': new_owner,
            'total_tokens': position['total_tokens'],
            'claimed_tokens': position['claimed_tokens']
        }
    
    def get_position(self, position_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve position details"""
        return self.positions.get(position_id)
    
    def get_user_positions(self, owner: str) -> list:
        """Get all positions for a user"""
        return [p for p in self.positions.values() if p['owner'] == owner]
