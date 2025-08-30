import typer
from typer import Context

from torusdk.cli._common import (
    make_custom_context,
)

curator_app = typer.Typer(no_args_is_help=True)


@curator_app.command()
def accept_application(
    ctx: Context,
    curator_key: str,
    application_id: int,
):
    """
    Accepts an application.
    """

    context = make_custom_context(ctx)
    client = context.com_client()

    resolved_curator_key = context.load_key(curator_key, None)

    with context.progress_status("Accepting application..."):
        client.accept_application(resolved_curator_key, application_id)
    context.info("Application accepted.")


@curator_app.command()
def add_to_whitelist(ctx: Context, curator_key: str, agent_key: str):
    """
    Adds an agent to a whitelist.
    """
    context = make_custom_context(ctx)
    client = context.com_client()

    resolved_curator_key = context.load_key(curator_key, None)
    resolved_agent_key = context.resolve_ss58(agent_key)

    with context.progress_status(f"Adding Agent {agent_key} to whitelist..."):
        client.add_to_whitelist(
            curator_key=resolved_curator_key, agent_key=resolved_agent_key
        )
    context.info(f"Agent {agent_key} added to whitelist")
