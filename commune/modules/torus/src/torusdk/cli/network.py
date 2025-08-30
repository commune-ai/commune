import typer
from typer import Context

from torusdk.cli._common import (
    make_custom_context,
    render_pydantic_table,
)
from torusdk.misc import (
    get_global_params,
)

network_app = typer.Typer(no_args_is_help=True)


@network_app.command()
def last_block(ctx: Context, hash: bool = False):
    """
    Gets the last block
    """
    context = make_custom_context(ctx)
    client = context.com_client()

    info = "number" if not hash else "hash"

    block = client.get_block()
    block_info = None
    if block:
        block_info = block["header"][info]

    context.output(str(block_info))


@network_app.command()
def params(ctx: Context):
    """
    Gets global params
    """
    context = make_custom_context(ctx)
    client = context.com_client()

    with context.progress_status("Getting global network params ..."):
        global_params = get_global_params(client)
    render_pydantic_table(global_params, context.console)
