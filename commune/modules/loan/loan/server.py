import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from solana.rpc.async_api import AsyncClient
from solana.keypair import Keypair
from solana.publickey import PublicKey
from anchorpy import Program, Provider, Wallet, Idl
from spl.token.async_client import AsyncToken
from spl.token.constants import TOKEN_PROGRAM_ID, ASSOCIATED_TOKEN_ACCOUNT_PROGRAM_ID
from spl.token.instructions import get_associated_token_address
import commune as c

class RetroLoanServer(c.Module):
    """
    🎮 RETRO P2P LOAN SERVER - TRUSTLESS SOLANA LENDING 🎮
    ╔══════════════════════════════════════════════════╗
    ║  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄       ║
    ║ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌      ║
    ║ ▐░█▀▀▀▀▀▀▀█░▌ ▀▀▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌      ║
    ║ ▐░▌       ▐░▌          ▐░▌▐░▌       ▐░▌      ║
    ║ ▐░█▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌      ║
    ║ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌      ║
    ║ ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀       ║
    ║ ▐░▌          ▐░▌          ▐░▌                ║
    ║ ▐░▌          ▐░█▄▄▄▄▄▄▄▄▄ ▐░▌                ║
    ║ ▐░▌          ▐░░░░░░░░░░░▌▐░▌                ║
    ║  ▀            ▀▀▀▀▀▀▀▀▀▀▀  ▀                 ║
    ╚══════════════════════════════════════════════════╝
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.networks = {
            'local': {
                'name': '[ LOCAL NET ]',
                'rpc': 'http://localhost:8545',
                'chain_id': 1337,
                'program_id': 'Loan111111111111111111111111111111111111111'
            },
            'testnet': {
                'name': '[ TEST NET ]',
                'rpc': 'https://api.testnet.solana.com',
                'chain_id': 11155111,
                'program_id': 'Loan111111111111111111111111111111111111111'
            },
            'mainnet': {
                'name': '[ MAIN NET ]',
                'rpc': 'https://api.mainnet-beta.solana.com',
                'chain_id': 1,
                'program_id': 'Loan111111111111111111111111111111111111111'
            }
        }
        self.current_network = None
        self.client = None
        self.program = None
        self.provider = None
        
    def print_retro(self, msg: str, color: str = 'green'):
        """Print messages in retro style"""
        border = "═" * (len(msg) + 4)
        c.print(f"╔{border}╗", color=color)
        c.print(f"║ {msg} ║", color=color)
        c.print(f"╚{border}╝", color=color)
        
    async def connect_network(self, network: str = 'testnet') -> Dict[str, Any]:
        """Connect to specified network"""
        try:
            self.print_retro(f"CONNECTING TO {self.networks[network]['name']}...", 'cyan')
            
            self.current_network = self.networks[network]
            self.client = AsyncClient(self.current_network['rpc'])
            
            # Load wallet (in production, load from secure storage)
            wallet = Wallet(Keypair())
            self.provider = Provider(self.client, wallet)
            
            # Load IDL and initialize program
            idl_path = os.path.join(os.path.dirname(__file__), 'idl', 'loan.json')
            if os.path.exists(idl_path):
                with open(idl_path) as f:
                    idl = Idl.from_json(f.read())
                program_id = PublicKey(self.current_network['program_id'])
                self.program = Program(idl, program_id, self.provider)
            
            self.print_retro(f"CONNECTED TO {self.current_network['name']}!", 'green')
            
            return {
                'success': True,
                'network': network,
                'rpc': self.current_network['rpc'],
                'wallet': str(wallet.public_key)
            }
            
        except Exception as e:
            self.print_retro(f"CONNECTION FAILED: {str(e)}", 'red')
            return {'success': False, 'error': str(e)}
    
    def derive_loan_pda(self, maker: PublicKey, mint_loan: PublicKey, mint_collat: PublicKey):
        """Derive loan PDA address"""
        seeds = [b"loan", bytes(maker), bytes(mint_loan), bytes(mint_collat)]
        return PublicKey.find_program_address(seeds, PublicKey(self.current_network['program_id']))
    
    async def create_offer(self, 
                          principal: int,
                          collateral: int,
                          interest_bps: int = 500,
                          duration_seconds: int = 604800,
                          mint_loan: str = None,
                          mint_collat: str = None) -> Dict[str, Any]:
        """Create a new loan offer"""
        try:
            self.print_retro("CREATING LOAN OFFER...", 'cyan')
            
            if not self.program:
                return {'success': False, 'error': 'Not connected to network'}
            
            maker = self.provider.wallet.public_key
            mint_loan_pk = PublicKey(mint_loan or "So11111111111111111111111111111111111111112")
            mint_collat_pk = PublicKey(mint_collat or "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
            
            # Derive PDAs
            loan_pda, bump = self.derive_loan_pda(maker, mint_loan_pk, mint_collat_pk)
            maker_loan_vault = get_associated_token_address(loan_pda, mint_loan_pk)
            
            # Create offer transaction
            tx = await self.program.rpc["create_offer"](
                principal, collateral, interest_bps, duration_seconds,
                ctx=self.program.context(
                    accounts={
                        "maker": maker,
                        "loan": loan_pda,
                        "makerLoanVault": maker_loan_vault,
                        "mintLoan": mint_loan_pk,
                        "mintCollat": mint_collat_pk,
                        "systemProgram": PublicKey("11111111111111111111111111111111"),
                        "tokenProgram": TOKEN_PROGRAM_ID,
                        "rent": PublicKey("SysvarRent111111111111111111111111111111111"),
                        "associatedTokenProgram": ASSOCIATED_TOKEN_ACCOUNT_PROGRAM_ID
                    }
                )
            )
            
            self.print_retro(f"OFFER CREATED! TX: {tx[:8]}...", 'green')
            
            return {
                'success': True,
                'tracker': tx,
                'loan_pda': str(loan_pda),
                'principal': principal,
                'collateral': collateral,
                'interest_bps': interest_bps,
                'duration': duration_seconds
            }
            
        except Exception as e:
            self.print_retro(f"OFFER CREATION FAILED: {str(e)}", 'red')
            return {'success': False, 'error': str(e)}
    
    async def accept_offer(self, loan_pda: str, maker: str) -> Dict[str, Any]:
        """Accept an existing loan offer"""
        try:
            self.print_retro("ACCEPTING LOAN OFFER...", 'cyan')
            
            if not self.program:
                return {'success': False, 'error': 'Not connected to network'}
            
            taker = self.provider.wallet.public_key
            loan_pda_pk = PublicKey(loan_pda)
            maker_pk = PublicKey(maker)
            
            # Fetch loan account to get mints
            loan_account = await self.program.account["Loan"].fetch(loan_pda_pk)
            
            # Derive all required accounts
            maker_loan_vault = get_associated_token_address(loan_pda_pk, loan_account.mint_loan)
            taker_loan_ata = get_associated_token_address(taker, loan_account.mint_loan)
            taker_collat_ata = get_associated_token_address(taker, loan_account.mint_collat)
            loan_collat_vault = get_associated_token_address(loan_pda_pk, loan_account.mint_collat)
            
            # Accept offer transaction
            tx = await self.program.rpc["accept_offer"](
                ctx=self.program.context(
                    accounts={
                        "taker": taker,
                        "maker": maker_pk,
                        "loan": loan_pda_pk,
                        "makerLoanVault": maker_loan_vault,
                        "takerLoanAta": taker_loan_ata,
                        "takerCollatAta": taker_collat_ata,
                        "loanCollatVault": loan_collat_vault,
                        "mintLoan": loan_account.mint_loan,
                        "mintCollat": loan_account.mint_collat,
                        "tokenProgram": TOKEN_PROGRAM_ID,
                        "associatedTokenProgram": ASSOCIATED_TOKEN_ACCOUNT_PROGRAM_ID,
                        "systemProgram": PublicKey("11111111111111111111111111111111"),
                        "clock": PublicKey("SysvarC1ock11111111111111111111111111111111")
                    }
                )
            )
            
            self.print_retro(f"OFFER ACCEPTED! TX: {tx[:8]}...", 'green')
            
            return {
                'success': True,
                'tracker': tx,
                'loan_pda': loan_pda,
                'taker': str(taker)
            }
            
        except Exception as e:
            self.print_retro(f"ACCEPT FAILED: {str(e)}", 'red')
            return {'success': False, 'error': str(e)}
    
    async def repay_loan(self, loan_pda: str) -> Dict[str, Any]:
        """Repay an active loan"""
        try:
            self.print_retro("REPAYING LOAN...", 'cyan')
            
            if not self.program:
                return {'success': False, 'error': 'Not connected to network'}
            
            taker = self.provider.wallet.public_key
            loan_pda_pk = PublicKey(loan_pda)
            
            # Fetch loan account
            loan_account = await self.program.account["Loan"].fetch(loan_pda_pk)
            
            # Derive required accounts
            taker_loan_ata = get_associated_token_address(taker, loan_account.mint_loan)
            maker_loan_vault = get_associated_token_address(loan_pda_pk, loan_account.mint_loan)
            loan_collat_vault = get_associated_token_address(loan_pda_pk, loan_account.mint_collat)
            taker_collat_ata = get_associated_token_address(taker, loan_account.mint_collat)
            
            # Repay transaction
            tx = await self.program.rpc["repay"](
                ctx=self.program.context(
                    accounts={
                        "taker": taker,
                        "maker": loan_account.maker,
                        "loan": loan_pda_pk,
                        "takerLoanAta": taker_loan_ata,
                        "makerLoanVault": maker_loan_vault,
                        "loanCollatVault": loan_collat_vault,
                        "takerCollatAta": taker_collat_ata,
                        "tokenProgram": TOKEN_PROGRAM_ID,
                        "clock": PublicKey("SysvarC1ock11111111111111111111111111111111")
                    }
                )
            )
            
            self.print_retro(f"LOAN REPAID! TX: {tx[:8]}...", 'green')
            
            return {
                'success': True,
                'tracker': tx,
                'loan_pda': loan_pda
            }
            
        except Exception as e:
            self.print_retro(f"REPAY FAILED: {str(e)}", 'red')
            return {'success': False, 'error': str(e)}
    
    async def liquidate_loan(self, loan_pda: str) -> Dict[str, Any]:
        """Liquidate an expired loan"""
        try:
            self.print_retro("LIQUIDATING LOAN...", 'cyan')
            
            if not self.program:
                return {'success': False, 'error': 'Not connected to network'}
            
            maker = self.provider.wallet.public_key
            loan_pda_pk = PublicKey(loan_pda)
            
            # Fetch loan account
            loan_account = await self.program.account["Loan"].fetch(loan_pda_pk)
            
            # Derive required accounts
            loan_collat_vault = get_associated_token_address(loan_pda_pk, loan_account.mint_collat)
            maker_collat_ata = get_associated_token_address(maker, loan_account.mint_collat)
            
            # Liquidate transaction
            tx = await self.program.rpc["liquidate"](
                ctx=self.program.context(
                    accounts={
                        "maker": maker,
                        "loan": loan_pda_pk,
                        "loanCollatVault": loan_collat_vault,
                        "makerCollatAta": maker_collat_ata,
                        "tokenProgram": TOKEN_PROGRAM_ID,
                        "clock": PublicKey("SysvarC1ock11111111111111111111111111111111")
                    }
                )
            )
            
            self.print_retro(f"LOAN LIQUIDATED! TX: {tx[:8]}...", 'green')
            
            return {
                'success': True,
                'tracker': tx,
                'loan_pda': loan_pda
            }
            
        except Exception as e:
            self.print_retro(f"LIQUIDATE FAILED: {str(e)}", 'red')
            return {'success': False, 'error': str(e)}
    
    async def get_loan_status(self, loan_pda: str) -> Dict[str, Any]:
        """Get current status of a loan"""
        try:
            if not self.program:
                return {'success': False, 'error': 'Not connected to network'}
            
            loan_pda_pk = PublicKey(loan_pda)
            loan_account = await self.program.account["Loan"].fetch(loan_pda_pk)
            
            # Calculate loan details
            is_active = loan_account.taker is not None
            current_time = datetime.now().timestamp()
            
            status = {
                'success': True,
                'loan_pda': loan_pda,
                'maker': str(loan_account.maker),
                'taker': str(loan_account.taker) if loan_account.taker else None,
                'principal': loan_account.principal,
                'collateral': loan_account.collateral_required,
                'interest_bps': loan_account.interest_bps,
                'duration': loan_account.duration_seconds,
                'is_active': is_active,
                'mint_loan': str(loan_account.mint_loan),
                'mint_collat': str(loan_account.mint_collat)
            }
            
            if is_active and loan_account.start_ts:
                expiry = loan_account.start_ts + loan_account.duration_seconds
                status['start_time'] = loan_account.start_ts
                status['expiry_time'] = expiry
                status['is_expired'] = current_time > expiry
                status['time_remaining'] = max(0, expiry - current_time)
            
            return status
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def list_active_loans(self, maker: Optional[str] = None, taker: Optional[str] = None) -> Dict[str, Any]:
        """List all active loans, optionally filtered by maker or taker"""
        try:
            if not self.program:
                return {'success': False, 'error': 'Not connected to network'}
            
            # Fetch all loan accounts
            filters = []
            if maker:
                filters.append({"memcmp": {"offset": 8, "bytes": maker}})
            elif taker:
                filters.append({"memcmp": {"offset": 40, "bytes": taker}})
            
            loans = await self.program.account["Loan"].all(filters=filters)
            
            loan_list = []
            for loan_account, loan_pubkey in loans:
                loan_info = await self.get_loan_status(str(loan_pubkey))
                if loan_info['success']:
                    loan_list.append(loan_info)
            
            return {
                'success': True,
                'count': len(loan_list),
                'loans': loan_list
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def forward(self, action: str = 'help', **kwargs) -> Dict[str, Any]:
        """Main entry point for the loan server"""
        self.print_retro("🎮 RETRO P2P LOAN SERVER 🎮", 'cyan')
        
        actions = {
            'connect': self.connect_network,
            'create_offer': self.create_offer,
            'accept_offer': self.accept_offer,
            'repay': self.repay_loan,
            'liquidate': self.liquidate_loan,
            'status': self.get_loan_status,
            'list': self.list_active_loans
        }
        
        if action == 'help':
            help_text = """
            ╔════════════════════════════════════════╗
            ║         AVAILABLE COMMANDS:            ║
            ╠════════════════════════════════════════╣
            ║ connect      - Connect to network      ║
            ║ create_offer - Create loan offer       ║
            ║ accept_offer - Accept loan offer       ║
            ║ repay        - Repay active loan       ║
            ║ liquidate    - Liquidate expired loan  ║
            ║ status       - Check loan status       ║
            ║ list         - List active loans       ║
            ╚════════════════════════════════════════╝
            """
            c.print(help_text, color='green')
            return {'success': True, 'message': 'Help displayed'}
        
        if action in actions:
            # Run async function
            return asyncio.run(actions[action](**kwargs))
        else:
            self.print_retro(f"UNKNOWN ACTION: {action}", 'red')
            return {'success': False, 'error': f'Unknown action: {action}'}

# Register the module
if __name__ == "__main__":
    server = RetroLoanServer()
    # Example usage
    result = server.forward('connect', network='testnet')
    c.print(result)