import time
from typing import Optional

import typer
from typer import Context

from torusdk.balance import BalanceUnit, format_balance, to_rems
from torusdk.cli._common import (
    make_custom_context,
    print_table_from_plain_dict,
)
from torusdk.errors import ChainTransactionError
from torusdk.faucet.powv2 import solve_for_difficulty_fast

balance_app = typer.Typer(no_args_is_help=True)


@balance_app.command()
def free_balance(
    ctx: Context,
    key: str,
    unit: BalanceUnit = BalanceUnit.joule,
):
    """
    Gets free balance of a key.
    """
    context = make_custom_context(ctx)
    client = context.com_client()

    key_address = context.resolve_ss58(key)

    with context.progress_status(
        f"Getting free balance of key {key_address}..."
    ):
        balance = client.get_balance(key_address)

    context.output(format_balance(balance, unit))


@balance_app.command()
def staked_balance(
    ctx: Context,
    key: str,
    unit: BalanceUnit = BalanceUnit.joule,
    password: Optional[str] = None,
):
    """
    Gets the balance staked on the key itself.
    """
    context = make_custom_context(ctx)
    client = context.com_client()

    key_address = context.resolve_ss58(key)

    with context.progress_status(
        f"Getting staked balance of key {key_address}..."
    ):
        result = sum(client.get_stakingto(key=key_address).values())

    context.output(format_balance(result, unit))


@balance_app.command()
def show(
    ctx: Context,
    key: str,
    unit: BalanceUnit = BalanceUnit.joule,
):
    """
    Gets entire balance of a key (free balance + staked balance).
    """
    context = make_custom_context(ctx)
    client = context.com_client()

    key_address = context.resolve_ss58(key)

    with context.progress_status(f"Getting value of key {key_address}..."):
        staked_balance = sum(client.get_stakingto(key=key_address).values())
        free_balance = client.get_balance(key_address)
        balance_sum = free_balance + staked_balance

    print_table_from_plain_dict(
        {
            "Free": format_balance(free_balance, unit),
            "Staked": format_balance(staked_balance, unit),
            "Total": format_balance(balance_sum, unit),
        },
        ["Result", "Amount"],
        context.console,
    )


@balance_app.command()
def get_staked(
    ctx: Context,
    key: str,
    unit: BalanceUnit = BalanceUnit.joule,
):
    """
    Gets total stake of a key it delegated across other keys.
    """
    context = make_custom_context(ctx)
    client = context.com_client()

    key_address = context.resolve_ss58(key)

    with context.progress_status(f"Getting stake of {key_address}..."):
        result = sum(client.get_stakingto(key=key_address).values())

    context.output(format_balance(result, unit))


@balance_app.command()
def transfer(ctx: Context, key: str, amount: float, dest: str):
    """
    Transfer amount to destination using key
    """
    context = make_custom_context(ctx)
    client = context.com_client()

    nano_amount = to_rems(amount)

    resolved_key = context.load_key(key, None)
    resolved_dest = context.resolve_ss58(dest)

    if not context.confirm(
        f"Are you sure you want to transfer {amount} tokens to {dest}?"
    ):
        raise typer.Abort()

    with context.progress_status(f"Transferring {amount} tokens to {dest}..."):
        response = client.transfer(
            key=resolved_key, amount=nano_amount, dest=resolved_dest
        )

    if response.is_success:
        context.info(f"Transferred {amount} tokens to {dest}")
    else:
        raise ChainTransactionError(response.error_message)  # type: ignore


@balance_app.command()
def transfer_stake(
    ctx: Context, key: str, amount: float, from_key: str, dest: str
):
    """
    Transfers stake of key from point A to point B
    """
    context = make_custom_context(ctx)
    client = context.com_client()

    nano_amount = to_rems(amount)
    keypair = context.load_key(key, None)
    resolved_from = context.resolve_ss58(from_key)
    resolved_dest = context.resolve_ss58(dest)

    with context.progress_status(
        f"Transferring {amount} tokens from {from_key} to {dest}' ..."
    ):
        response = client.transfer_stake(
            key=keypair,
            amount=nano_amount,
            from_module_key=resolved_from,
            dest_module_address=resolved_dest,
        )

    if response.is_success:
        context.info(f"Transferred {amount} tokens from {from_key} to {dest}")
    else:
        raise ChainTransactionError(response.error_message)  # type: ignore


@balance_app.command()
def stake(
    ctx: Context,
    key: str,
    amount: float,
    dest: str,
):
    """
    Stake amount to destination using key
    """
    context = make_custom_context(ctx)
    client = context.com_client()

    nano_amount = to_rems(amount)
    keypair = context.load_key(key, None)
    resolved_dest = context.resolve_ss58(dest)

    delegating_message = (
        "By default you delegate DAO "
        "voting power to the validator you stake to. "
        "In case you want to change this, call: "
        "`torus-cli key power-delegation <key> --disable`."
    )
    context.info("INFO: ", style="bold green", end="")  # type: ignore
    context.info(delegating_message)  # type: ignore
    with context.progress_status(f"Staking {amount} tokens to {dest}..."):
        response = client.stake(
            key=keypair, amount=nano_amount, dest=resolved_dest
        )

    if response.is_success:
        context.info(f"Staked {amount} tokens to {dest}")
    else:
        raise ChainTransactionError(response.error_message)  # type: ignore


@balance_app.command()
def unstake(ctx: Context, key: str, amount: float, dest: str):
    """
    Unstake amount from destination using key
    """
    context = make_custom_context(ctx)
    client = context.com_client()

    nano_amount = to_rems(amount)
    keypair = context.load_key(key, None)
    resolved_dest = context.resolve_ss58(dest)

    with context.progress_status(f"Unstaking {amount} tokens from {dest}'..."):
        response = client.unstake(
            key=keypair, amount=nano_amount, dest=resolved_dest
        )

    if response.is_success:
        context.info(f"Unstaked {amount} tokens from {dest}")
    else:
        raise ChainTransactionError(response.error_message)  # type: ignore


# Ammount of seconds to wait between faucet executions
SLEEP_BETWEEN_FAUCET_EXECUTIONS = 8


@balance_app.command()
def run_faucet(
    ctx: Context,
    key: str,
    jobs: Optional[int] = None,
    repeat: int = 1,
):
    context = make_custom_context(ctx)
    use_testnet = ctx.obj.use_testnet

    if not use_testnet:
        context.error("Faucet only enabled on testnet")
        raise typer.Exit(code=1)

    resolved_key = context.load_key(key, None)

    client = context.com_client()
    for _i in range(repeat):
        with context.progress_status("Solving PoW..."):
            solution = solve_for_difficulty_fast(
                client,
                resolved_key,
                client.url,
                num_processes=jobs,
            )
        with context.progress_status("Sending solution to blockchain"):
            params = {
                "block_number": solution.block_number,
                "nonce": solution.nonce,
                "work": solution.seal,
                "key": resolved_key.ss58_address,
            }

            client.compose_call(
                "faucet",
                params=params,
                unsigned=True,
                module="Faucet",
                key=resolved_key,
                wait_for_inclusion=False,
            )

        context.info(
            f"Waiting {SLEEP_BETWEEN_FAUCET_EXECUTIONS} seconds before next execution..."
        )
        time.sleep(SLEEP_BETWEEN_FAUCET_EXECUTIONS)
