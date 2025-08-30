import struct
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from solana.rpc.commitment import Processed
from solana.rpc.types import MemcmpOpts
from solders.instruction import AccountMeta, Instruction  # type: ignore
from solders.pubkey import Pubkey  # type: ignore

from config import client
from layouts.amm_v4 import LIQUIDITY_STATE_LAYOUT_V4, MARKET_STATE_LAYOUT_V3
from layouts.cpmm import CPMM_POOL_STATE_LAYOUT
from raydium.constants import (
    WSOL,
    RAYDIUM_AMM_V4,
    RAYDIUM_CPMM,
    DEFAULT_QUOTE_MINT,
)

@dataclass
class AmmV4PoolKeys:
    amm_id: Pubkey
    base_mint: Pubkey
    quote_mint: Pubkey
    base_decimals: int
    quote_decimals: int
    open_orders: Pubkey
    target_orders: Pubkey
    base_vault: Pubkey
    quote_vault: Pubkey
    market_id: Pubkey
    market_authority: Pubkey
    market_base_vault: Pubkey
    market_quote_vault: Pubkey
    bids: Pubkey
    asks: Pubkey
    event_queue: Pubkey
    ray_authority_v4: Pubkey
    open_book_program: Pubkey
    token_program_id: Pubkey

@dataclass
class CpmmPoolKeys:
    pool_state: Pubkey
    raydium_vault_auth_2: Pubkey
    amm_config: Pubkey
    pool_creator: Pubkey
    token_0_vault: Pubkey
    token_1_vault: Pubkey
    lp_mint: Pubkey
    token_0_mint: Pubkey
    token_1_mint: Pubkey
    token_0_program: Pubkey
    token_1_program: Pubkey
    observation_key: Pubkey
    auth_bump: int
    status: int
    lp_mint_decimals: int
    mint_0_decimals: int
    mint_1_decimals: int
    lp_supply: int
    protocol_fees_token_0: int
    protocol_fees_token_1: int
    fund_fees_token_0: int
    fund_fees_token_1: int
    open_time: int

class DIRECTION(Enum):
    BUY = 0
    SELL = 1

def fetch_amm_v4_pool_keys(pair_address: str) -> Optional[AmmV4PoolKeys]:
    
    def bytes_of(value):
        if not (0 <= value < 2**64):
            raise ValueError("Value must be in the range of a u64 (0 to 2^64 - 1).")
        return struct.pack('<Q', value)
   
    try:
        amm_id = Pubkey.from_string(pair_address)
        amm_data = client.get_account_info_json_parsed(amm_id, commitment=Processed).value.data
        amm_data_decoded = LIQUIDITY_STATE_LAYOUT_V4.parse(amm_data)
        marketId = Pubkey.from_bytes(amm_data_decoded.serumMarket)
        marketInfo = client.get_account_info_json_parsed(marketId, commitment=Processed).value.data
        market_decoded = MARKET_STATE_LAYOUT_V3.parse(marketInfo)
        vault_signer_nonce = market_decoded.vault_signer_nonce
        
        ray_authority_v4=Pubkey.from_string("5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1")
        open_book_program=Pubkey.from_string("srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX")
        token_program_id=Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

        pool_keys = AmmV4PoolKeys(
            amm_id=amm_id,
            base_mint=Pubkey.from_bytes(market_decoded.base_mint),
            quote_mint=Pubkey.from_bytes(market_decoded.quote_mint),
            base_decimals=amm_data_decoded.coinDecimals,
            quote_decimals=amm_data_decoded.pcDecimals,
            open_orders=Pubkey.from_bytes(amm_data_decoded.ammOpenOrders),
            target_orders=Pubkey.from_bytes(amm_data_decoded.ammTargetOrders),
            base_vault=Pubkey.from_bytes(amm_data_decoded.poolCoinTokenAccount),
            quote_vault=Pubkey.from_bytes(amm_data_decoded.poolPcTokenAccount),
            market_id=marketId,
            market_authority=Pubkey.create_program_address(seeds=[bytes(marketId), bytes_of(vault_signer_nonce)], program_id=open_book_program),
            market_base_vault=Pubkey.from_bytes(market_decoded.base_vault),
            market_quote_vault=Pubkey.from_bytes(market_decoded.quote_vault),
            bids=Pubkey.from_bytes(market_decoded.bids),
            asks=Pubkey.from_bytes(market_decoded.asks),
            event_queue=Pubkey.from_bytes(market_decoded.event_queue),
            ray_authority_v4=ray_authority_v4,
            open_book_program=open_book_program,
            token_program_id=token_program_id
        )

        return pool_keys
    except Exception as e:
        print(f"Error fetching AMMv4 pool keys: {e}")
        return None

def fetch_cpmm_pool_keys(pair_address: str) -> Optional[CpmmPoolKeys]:
    try:
        pool_state = Pubkey.from_string(pair_address)
        raydium_vault_auth_2 = Pubkey.from_string("GpMZbSM2GgvTKHJirzeGfMFoaZ8UR2X7F4v8vHTvxFbL")
        pool_state_data = client.get_account_info_json_parsed(pool_state, commitment=Processed).value.data
        parsed_data = CPMM_POOL_STATE_LAYOUT.parse(pool_state_data)

        pool_keys = CpmmPoolKeys(
            pool_state=pool_state,
            raydium_vault_auth_2 = raydium_vault_auth_2,
            amm_config=Pubkey.from_bytes(parsed_data.amm_config),
            pool_creator=Pubkey.from_bytes(parsed_data.pool_creator),
            token_0_vault=Pubkey.from_bytes(parsed_data.token_0_vault),
            token_1_vault=Pubkey.from_bytes(parsed_data.token_1_vault),
            lp_mint=Pubkey.from_bytes(parsed_data.lp_mint),
            token_0_mint=Pubkey.from_bytes(parsed_data.token_0_mint),
            token_1_mint=Pubkey.from_bytes(parsed_data.token_1_mint),
            token_0_program=Pubkey.from_bytes(parsed_data.token_0_program),
            token_1_program=Pubkey.from_bytes(parsed_data.token_1_program),
            observation_key=Pubkey.from_bytes(parsed_data.observation_key),
            auth_bump=parsed_data.auth_bump,
            status=parsed_data.status,
            lp_mint_decimals=parsed_data.lp_mint_decimals,
            mint_0_decimals=parsed_data.mint_0_decimals,
            mint_1_decimals=parsed_data.mint_1_decimals,
            lp_supply=parsed_data.lp_supply,
            protocol_fees_token_0=parsed_data.protocol_fees_token_0,
            protocol_fees_token_1=parsed_data.protocol_fees_token_1,
            fund_fees_token_0=parsed_data.fund_fees_token_0,
            fund_fees_token_1=parsed_data.fund_fees_token_1,
            open_time=parsed_data.open_time,
        )
        
        return pool_keys
    
    except Exception as e:
        print(f"Error fetching CPMM pool keys: {e}")
        return None

def make_amm_v4_swap_instruction(
    amount_in: int, 
    minimum_amount_out: int, 
    token_account_in: Pubkey, 
    token_account_out: Pubkey, 
    accounts: AmmV4PoolKeys,
    owner: Pubkey
) -> Instruction:
    try:
        
        keys = [
            AccountMeta(pubkey=accounts.token_program_id, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.amm_id, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.ray_authority_v4, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.open_orders, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.target_orders, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.base_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.quote_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.open_book_program, is_signer=False, is_writable=False), 
            AccountMeta(pubkey=accounts.market_id, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.bids, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.asks, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.event_queue, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.market_base_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.market_quote_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.market_authority, is_signer=False, is_writable=False),
            AccountMeta(pubkey=token_account_in, is_signer=False, is_writable=True),  
            AccountMeta(pubkey=token_account_out, is_signer=False, is_writable=True), 
            AccountMeta(pubkey=owner, is_signer=True, is_writable=False) 
        ]
        
        data = bytearray()
        discriminator = 9
        data.extend(struct.pack('<B', discriminator))
        data.extend(struct.pack('<Q', amount_in))
        data.extend(struct.pack('<Q', minimum_amount_out))
        swap_instruction = Instruction(RAYDIUM_AMM_V4, bytes(data), keys)
        
        return swap_instruction
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def make_cpmm_swap_instruction(
    amount_in: int,
    minimum_amount_out: int,
    token_account_in: Pubkey,
    token_account_out: Pubkey,
    accounts: CpmmPoolKeys,
    owner: Pubkey,
    action: DIRECTION,
) -> Instruction:

    try:
        if action == DIRECTION.BUY:
            input_vault = accounts.token_0_vault
            output_vault = accounts.token_1_vault
            input_token_program = accounts.token_0_program
            output_token_program = accounts.token_1_program
            input_token_mint = accounts.token_0_mint
            output_token_mint = accounts.token_1_mint
        else:  # SELL
            input_vault = accounts.token_1_vault
            output_vault = accounts.token_0_vault
            input_token_program = accounts.token_1_program
            output_token_program = accounts.token_0_program
            input_token_mint = accounts.token_1_mint
            output_token_mint = accounts.token_0_mint

        keys = [
            AccountMeta(pubkey=owner, is_signer=True, is_writable=True),
            AccountMeta(pubkey=accounts.raydium_vault_auth_2, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.amm_config, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.pool_state, is_signer=False, is_writable=True),
            AccountMeta(pubkey=token_account_in, is_signer=False, is_writable=True),
            AccountMeta(pubkey=token_account_out, is_signer=False, is_writable=True),
            AccountMeta(pubkey=input_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=output_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=input_token_program, is_signer=False, is_writable=False),
            AccountMeta(pubkey=output_token_program, is_signer=False, is_writable=False),
            AccountMeta(pubkey=input_token_mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=output_token_mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.observation_key, is_signer=False, is_writable=True),
        ]

        data = bytearray()
        data.extend(bytes.fromhex("8fbe5adac41e33de"))
        data.extend(struct.pack("<Q", amount_in))
        data.extend(struct.pack("<Q", minimum_amount_out))

        swap_instruction = Instruction(RAYDIUM_CPMM, bytes(data), keys)
        return swap_instruction
    except Exception as e:
        print(f"Error occurred creating CPMM swap instruction: {e}")
        return None

def get_amm_v4_reserves(pool_keys: AmmV4PoolKeys) -> tuple:
    try:
        quote_vault = pool_keys.quote_vault
        quote_decimal = pool_keys.quote_decimals
        quote_mint = pool_keys.quote_mint
        
        base_vault = pool_keys.base_vault
        base_decimal = pool_keys.base_decimals
        base_mint = pool_keys.base_mint
        
        balances_response = client.get_multiple_accounts_json_parsed(
            [quote_vault, base_vault], 
            Processed
        )
        balances = balances_response.value

        quote_account = balances[0]
        base_account = balances[1]
        
        quote_account_balance = quote_account.data.parsed['info']['tokenAmount']['uiAmount']
        base_account_balance = base_account.data.parsed['info']['tokenAmount']['uiAmount']
        
        if quote_account_balance is None or base_account_balance is None:
            print("Error: One of the account balances is None.")
            return None, None, None
        
        if base_mint == WSOL:
            base_reserve = quote_account_balance  
            quote_reserve = base_account_balance  
            token_decimal = quote_decimal 
        else:
            base_reserve = base_account_balance  
            quote_reserve = quote_account_balance
            token_decimal = base_decimal

        print(f"Base Mint: {base_mint} | Quote Mint: {quote_mint}")
        print(f"Base Reserve: {base_reserve} | Quote Reserve: {quote_reserve} | Token Decimal: {token_decimal}")
        return base_reserve, quote_reserve, token_decimal

    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None, None

def get_cpmm_reserves(pool_keys: CpmmPoolKeys):
    quote_vault = pool_keys.token_0_vault
    quote_decimal = pool_keys.mint_0_decimals
    quote_mint = pool_keys.token_0_mint
    
    base_vault = pool_keys.token_1_vault
    base_decimal = pool_keys.mint_1_decimals
    base_mint = pool_keys.token_1_mint
    
    protocol_fees_token_0 = pool_keys.protocol_fees_token_0 / (10 ** quote_decimal)
    fund_fees_token_0 = pool_keys.fund_fees_token_0 / (10 ** quote_decimal)
    protocol_fees_token_1 = pool_keys.protocol_fees_token_1 / (10 ** base_decimal)
    fund_fees_token_1 = pool_keys.fund_fees_token_1 / (10 ** base_decimal)
    
    balances_response = client.get_multiple_accounts_json_parsed(
        [quote_vault, base_vault], 
        Processed
    )
    balances = balances_response.value

    quote_account = balances[0]
    base_account = balances[1]
    quote_account_balance = quote_account.data.parsed['info']['tokenAmount']['uiAmount']
    base_account_balance = base_account.data.parsed['info']['tokenAmount']['uiAmount']
    
    if quote_account_balance is None or base_account_balance is None:
        print("Error: One of the account balances is None.")
        return None, None, None
    
    if base_mint == WSOL:
        base_reserve = quote_account_balance - (protocol_fees_token_0 + fund_fees_token_0) 
        quote_reserve = base_account_balance - (protocol_fees_token_1 + fund_fees_token_1)
        token_decimal = quote_decimal
    else:
        base_reserve = base_account_balance - (protocol_fees_token_1 + fund_fees_token_1)
        quote_reserve = quote_account_balance - (protocol_fees_token_0 + fund_fees_token_0)
        token_decimal = base_decimal

    print(f"Base Mint: {base_mint} | Quote Mint: {quote_mint}")
    print(f"Base Reserve: {base_reserve} | Quote Reserve: {quote_reserve} | Token Decimal: {token_decimal}")
    return base_reserve, quote_reserve, token_decimal

def fetch_pair_address_from_rpc(
    program_id: Pubkey, 
    token_mint: str, 
    quote_offset: int, 
    base_offset: int, 
    data_length: int
) -> list:

    def fetch_pair(base_mint: str, quote_mint: str) -> list:
        memcmp_filter_base = MemcmpOpts(offset=quote_offset, bytes=quote_mint)
        memcmp_filter_quote = MemcmpOpts(offset=base_offset, bytes=base_mint)
        try:
            print(f"Fetching pair addresses for base_mint: {base_mint}, quote_mint: {quote_mint}")
            response = client.get_program_accounts(
                program_id,
                commitment=Processed,
                filters=[data_length, memcmp_filter_base, memcmp_filter_quote],
            )
            accounts = response.value
            if accounts:
                print(f"Found {len(accounts)} matching account(s).")
                return [account.pubkey.__str__() for account in accounts]
            else:
                print("No matching accounts found.")
        except Exception as e:
            print(f"Error fetching pair addresses: {e}")
        return []

    pair_addresses = fetch_pair(token_mint, DEFAULT_QUOTE_MINT)

    if not pair_addresses:
        print("Retrying with reversed base and quote mints...")
        pair_addresses = fetch_pair(DEFAULT_QUOTE_MINT, token_mint)

    return pair_addresses

def get_amm_v4_pair_from_rpc(token_mint: str) -> list:
    return fetch_pair_address_from_rpc(
        program_id=RAYDIUM_AMM_V4,
        token_mint=token_mint,
        quote_offset=400,
        base_offset=432,
        data_length=752,
    )

def get_cpmm_pair_address_from_rpc(token_mint: str) -> list:
    return fetch_pair_address_from_rpc(
        program_id=RAYDIUM_CPMM,
        token_mint=token_mint,
        quote_offset=168,
        base_offset=200,
        data_length=637,
    )
