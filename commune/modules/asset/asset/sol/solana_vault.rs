use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};

declare_id!("Fg6PaFpoGXkYsidMpWTK6W2BeZ7FEfcYkg476zPFsLnS");

#[program]
pub mod solana_vault {
    use super::*;

    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        let vault = &mut ctx.accounts.vault;
        vault.owner = ctx.accounts.owner.key();
        vault.total_deposits = 0;
        Ok(())
    }

    pub fn add_accepted_token(ctx: Context<AddToken>, token_mint: Pubkey) -> Result<()> {
        let vault = &mut ctx.accounts.vault;
        require!(vault.owner == ctx.accounts.owner.key(), VaultError::Unauthorized);
        
        if !vault.accepted_tokens.contains(&token_mint) {
            vault.accepted_tokens.push(token_mint);
        }
        Ok(())
    }

    pub fn deposit(ctx: Context<Deposit>, amount: u64) -> Result<()> {
        let vault = &ctx.accounts.vault;
        let token_mint = ctx.accounts.user_token.mint;
        
        require!(
            vault.accepted_tokens.contains(&token_mint),
            VaultError::TokenNotAccepted
        );
        require!(amount > 0, VaultError::InvalidAmount);

        // Transfer tokens from user to vault
        let cpi_accounts = Transfer {
            from: ctx.accounts.user_token.to_account_info(),
            to: ctx.accounts.vault_token.to_account_info(),
            authority: ctx.accounts.user.to_account_info(),
        };
        let cpi_program = ctx.accounts.token_program.to_account_info();
        let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
        token::transfer(cpi_ctx, amount)?;

        // Update user balance
        let user_account = &mut ctx.accounts.user_account;
        user_account.balances.push(TokenBalance {
            token_mint,
            amount,
        });

        Ok(())
    }

    pub fn withdraw(ctx: Context<Withdraw>, token_mint: Pubkey, amount: u64) -> Result<()> {
        let user_account = &mut ctx.accounts.user_account;
        
        // Find and update user balance
        let balance_index = user_account
            .balances
            .iter()
            .position(|b| b.token_mint == token_mint)
            .ok_or(VaultError::InsufficientBalance)?;
        
        require!(
            user_account.balances[balance_index].amount >= amount,
            VaultError::InsufficientBalance
        );

        user_account.balances[balance_index].amount -= amount;

        // Transfer tokens from vault to user
        let vault_key = ctx.accounts.vault.key();
        let seeds = &[b"vault", vault_key.as_ref(), &[ctx.bumps.vault]];
        let signer = &[&seeds[..]];
        
        let cpi_accounts = Transfer {
            from: ctx.accounts.vault_token.to_account_info(),
            to: ctx.accounts.user_token.to_account_info(),
            authority: ctx.accounts.vault.to_account_info(),
        };
        let cpi_program = ctx.accounts.token_program.to_account_info();
        let cpi_ctx = CpiContext::new_with_signer(cpi_program, cpi_accounts, signer);
        token::transfer(cpi_ctx, amount)?;

        Ok(())
    }
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(init, payer = owner, space = 8 + 32 + 8 + 32 * 10)]
    pub vault: Account<'info, Vault>,
    #[account(mut)]
    pub owner: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct AddToken<'info> {
    #[account(mut)]
    pub vault: Account<'info, Vault>,
    pub owner: Signer<'info>,
}

#[derive(Accounts)]
pub struct Deposit<'info> {
    #[account(mut)]
    pub vault: Account<'info, Vault>,
    #[account(
        init_if_needed,
        payer = user,
        space = 8 + 32 + 32 * 10,
        seeds = [b"user", user.key().as_ref()],
        bump
    )]
    pub user_account: Account<'info, UserAccount>,
    #[account(mut)]
    pub user_token: Account<'info, TokenAccount>,
    #[account(mut)]
    pub vault_token: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user: Signer<'info>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct Withdraw<'info> {
    #[account(mut)]
    pub vault: Account<'info, Vault>,
    #[account(mut)]
    pub user_account: Account<'info, UserAccount>,
    #[account(mut)]
    pub user_token: Account<'info, TokenAccount>,
    #[account(mut)]
    pub vault_token: Account<'info, TokenAccount>,
    pub user: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[account]
pub struct Vault {
    pub owner: Pubkey,
    pub total_deposits: u64,
    pub accepted_tokens: Vec<Pubkey>,
}

#[account]
pub struct UserAccount {
    pub user: Pubkey,
    pub balances: Vec<TokenBalance>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct TokenBalance {
    pub token_mint: Pubkey,
    pub amount: u64,
}

#[error_code]
pub enum VaultError {
    #[msg("Unauthorized")]
    Unauthorized,
    #[msg("Token not accepted")]
    TokenNotAccepted,
    #[msg("Invalid amount")]
    InvalidAmount,
    #[msg("Insufficient balance")]
    InsufficientBalance,
}