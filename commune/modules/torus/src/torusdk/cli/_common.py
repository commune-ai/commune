import os
import re
from dataclasses import dataclass
from getpass import getpass
from typing import Any, Callable, Mapping, TypeVar, cast

import rich
import rich.prompt
import typer
from pydantic import BaseModel
from rich import box
from rich.console import Console
from rich.table import Table
from torustrateinterface import Keypair
from typer import Context

from torusdk._common import CID_REGEX, TorusSettings, get_node_url
from torusdk.balance import dict_from_nano, from_rems, to_rems
from torusdk.client import TorusClient
from torusdk.errors import InvalidPasswordError, PasswordNotProvidedError
from torusdk.key import (
    is_ss58_address,
    key_path,
    load_keypair,
    resolve_key_ss58,
)
from torusdk.types.types import (
    AgentInfoWithOptionalBalance,
    Ss58Address,
)

NOT_IMPLEMENTED_MESSAGE = (
    "method not available. "
    "It's going to be rolled out in the coming updates."
)

HIDE_FEATURES = False


KEY_DEPRECATION_WARNING = (
    "\nYou are using a legacy key storage. "
    "This will be deprecated in the future. "
    "Please migrate your key to torus storage "
    "using the `torus key migrate [key]` command.\n"
)
KEY_DEPRECATION_STYLE = f"{typer.colors.RED} bold on yellow"


T = TypeVar("T", bound=BaseModel)


def merge_models(model_a: T, model_b: BaseModel) -> T:
    dict_a = model_a.model_dump()
    dict_b = model_b.model_dump(exclude_unset=True)
    unoptional_dict = {
        key: value for key, value in dict_b.items() if value is not None
    }
    merged_dict = {**dict_a, **unoptional_dict}
    return model_a.__class__(**merged_dict)


def extract_cid(value: str):
    cid_hash = re.match(CID_REGEX, value)
    if not cid_hash:
        raise typer.BadParameter(f"CID provided is invalid: {value}")
    return cid_hash.group("cid")


def input_to_rems(value: float | None):
    if value is None:
        return None
    return to_rems(value)


def check_storage_exists(console: Console):
    root_path = key_path("").replace(".json", "")
    if not os.path.exists(root_path):
        console.print(
            "Torus storage not found. Did you run `torus key migrate` "
            "after updating your package?",
            style="bold red",
        )
        raise typer.Exit(code=1)


@dataclass
class ExtraCtxData:
    output_json: bool
    use_testnet: bool
    yes_to_all: bool


class ExtendedContext(Context):
    obj: ExtraCtxData


class CliPasswordProvider:
    def __init__(
        self, settings: TorusSettings, prompt_secret: Callable[[str], str]
    ):
        self.settings = settings
        self.prompt_secret = prompt_secret

    def get_password(self, key_name: str) -> str | None:
        key_map = self.settings.KEY_PASSWORDS
        if key_map is not None:
            password = key_map.get(key_name)
            if password is not None:
                return password.get_secret_value()
        # fallback to universal password
        password = self.settings.UNIVERSAL_PASSWORD
        if password is not None:
            return password.get_secret_value()
        else:
            return None

    def ask_password(self, key_name: str) -> str:
        password = self.prompt_secret(
            f"Please provide the password for the key '{key_name}'"
        )
        return password


class CustomCtx:
    ctx: ExtendedContext
    settings: TorusSettings
    console: rich.console.Console
    console_err: rich.console.Console
    password_manager: CliPasswordProvider
    _com_client: TorusClient | None = None

    def __init__(
        self,
        ctx: ExtendedContext,
        settings: TorusSettings,
        console: rich.console.Console,
        console_err: rich.console.Console,
        com_client: TorusClient | None = None,
    ):
        self.ctx = ctx
        self.settings = settings
        self.console = console
        self.console_err = console_err
        self._com_client = com_client
        self.password_manager = CliPasswordProvider(
            self.settings, self.prompt_secret
        )

    def get_use_testnet(self) -> bool:
        return self.ctx.obj.use_testnet

    def get_node_url(self) -> str:
        use_testnet = self.get_use_testnet()
        return get_node_url(self.settings, use_testnet=use_testnet)

    def com_client(self) -> TorusClient:
        if self._com_client is None:
            node_url = self.get_node_url()
            self.info(f"Using node: {node_url}")
            for _ in range(5):
                try:
                    self._com_client = TorusClient(
                        url=node_url,
                        num_connections=1,
                        wait_for_finalization=False,
                        timeout=65,
                    )
                except Exception:
                    self.info(f"Failed to connect to node: {node_url}")
                    node_url = self.get_node_url()
                    self.info(f"Will retry with node {node_url}")
                    continue
                else:
                    break
            if self._com_client is None:
                raise ConnectionError("Could not connect to any node")

        return self._com_client

    def output(
        self,
        message: str,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> None:
        self.console.print(message, *args, **kwargs)  # type: ignore

    def info(
        self,
        message: str,
        *args: tuple[Any, ...],
        **kwargs: Any,
    ) -> None:
        self.console_err.print(message, *args, **kwargs)

    def error(
        self,
        message: str,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> None:
        message = f"ERROR: {message}"
        self.console_err.print(message, *args, style="bold red", **kwargs)  # type: ignore

    def progress_status(self, message: str):
        return self.console_err.status(message)

    def confirm(self, message: str) -> bool:
        if self.ctx.obj.yes_to_all:
            print(f"{message} (--yes)")
            return True
        return typer.confirm(message, err=True)

    def prompt_secret(self, message: str) -> str:
        return rich.prompt.Prompt.ask(
            message, password=True, console=self.console_err
        )

    def resolve_ss58(self, key: Ss58Address | Keypair | str):
        if isinstance(key, str) and not is_ss58_address(key):
            check_storage_exists(self.console_err)
        try:
            ss58 = resolve_key_ss58(key)
            return ss58
        except ValueError as e:
            self.error(e.args[0])
            raise typer.Exit(code=1)

    def load_key(self, key: str, password: str | None = None) -> Keypair:
        root_path = key_path("").replace(".json", "")
        if not os.path.exists(root_path):
            check_storage_exists(self.console_err)
        try:
            keypair = load_keypair(
                key, password, password_provider=self.password_manager
            )
            return keypair
        except PasswordNotProvidedError:
            self.error(f"Password not provided for key '{key}'")
            raise typer.Exit(code=1)
        except InvalidPasswordError:
            self.error(f"Incorrect password for key '{key}'")
            raise typer.Exit(code=1)


def make_custom_context(ctx: typer.Context) -> CustomCtx:
    return CustomCtx(
        ctx=cast(ExtendedContext, ctx),  # TODO: better check
        settings=TorusSettings(),
        console=Console(),
        console_err=Console(stderr=True),
    )


# Formatting


def eprint(e: Any) -> None:
    """
    Pretty prints an error.
    """

    console = Console()

    console.print(f"[bold red]ERROR: {e}", style="italic")


def print_table_from_plain_dict(
    result: Mapping[str, str | int | float | dict[Any, Any] | Ss58Address],
    column_names: list[str],
    console: Console,
) -> None:
    """
    Creates a table for a plain dictionary.
    """

    table = Table(show_header=True, header_style="bold magenta")

    for name in column_names:
        table.add_column(name, style="white", vertical="middle")

    # Add non-dictionary values to the table first
    for key, value in result.items():
        if not isinstance(value, dict):
            table.add_row(key, str(value))
    # Add subtables for nested dictionaries.
    # Important to add after so that the display of the table is nicer.
    for key, value in result.items():
        if isinstance(value, dict):
            subtable = Table(
                show_header=False,
                padding=(0, 0, 0, 0),
                border_style="bright_black",
            )
            for subkey, subvalue in value.items():
                subtable.add_row(f"{subkey}: {subvalue}")
            table.add_row(key, subtable)

    console.print(table)


def render_pydantic_subtable(value: BaseModel | dict[Any, Any]) -> Table:
    """
    Renders a subtable for a nested Pydantic object or dictionary.

    Args:
        value: A nested Pydantic object or dictionary.

    Returns:
        A rich Table object representing the subtable.
    """
    subtable = Table(
        show_header=False,
        padding=(0, 0, 0, 0),
        border_style="bright_black",
    )
    if isinstance(value, BaseModel):
        for subfield_name, _ in value.model_fields.items():
            subfield_value = getattr(value, subfield_name)
            subtable.add_row(f"{subfield_name}: {subfield_value}")
    else:
        for subfield_name, subfield_value in value.items():  # type: ignore
            subtable.add_row(f"{subfield_name}: {subfield_value}")
    return subtable


def render_single_pydantic_object(
    obj: BaseModel, console: Console, title: str = ""
) -> None:
    """
    Renders a rich table from a single Pydantic object.

    Args:
        obj: A single Pydantic object.
        console: The rich Console object.
        title: Optional title for the table.
    """
    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta",
        title_style="bold magenta",
    )

    table.add_column("Field", style="white", vertical="middle")
    table.add_column("Value", style="white", vertical="middle")

    for field_name, _ in obj.model_fields.items():
        value = getattr(obj, field_name)
        if isinstance(value, BaseModel):
            subtable = render_pydantic_subtable(value)
            table.add_row(field_name, subtable)
        else:
            table.add_row(field_name, str(value))

    console.print(table)
    console.print("\n")


def render_pydantic_table(
    objects: T | list[T],
    console: Console,
    title: str = "",
    ignored_columns: list[str] = [],
) -> None:
    """
    Renders a rich table from a list of Pydantic objects or a single Pydantic object.

    Args:
        objects: A list of Pydantic objects or a single Pydantic object.
        console: The rich Console object.
        title: Optional title for the table.
        ignored_columns: List of column names to ignore.
    """
    if not objects:
        return

    if isinstance(objects, BaseModel):
        render_single_pydantic_object(objects, console, title)
        return

    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta",
        title_style="bold magenta",
    )

    for field_name, _ in objects[0].model_fields.items():
        if field_name in ignored_columns:
            continue
        table.add_column(field_name, style="white", vertical="middle")

    for obj in objects:
        row_data: list[str | Table] = []
        for field_name, _ in obj.model_fields.items():
            if field_name in ignored_columns:
                continue
            value = getattr(obj, field_name)
            if isinstance(value, BaseModel):
                subtable = render_pydantic_subtable(value)
                row_data.append(subtable)
            else:
                row_data.append(str(value))
        table.add_row(*row_data)

    console.print(table)
    console.print("\n")


def print_table_standardize(
    result: dict[str, list[Any]], console: Console
) -> None:
    """
    Creates a table for a standardized dictionary.
    """
    table = Table(show_header=True, header_style="bold magenta")

    for key in result.keys():
        table.add_column(key, style="white")
    rows = [*result.values()]
    zipped_rows = [list(column) for column in zip(*rows)]
    for row in zipped_rows:
        table.add_row(*row, style="white")

    console.print(table)


def transform_module_into(
    to_exclude: list[str],
    last_block: int,
    immunity_period: int,
    agents: list[AgentInfoWithOptionalBalance],
):
    mods = cast(list[dict[str, Any]], agents)
    transformed_modules: list[dict[str, Any]] = []
    for mod in mods:
        module = mod.copy()
        module_regblock = module["regblock"]
        module["in_immunity"] = module_regblock + immunity_period > last_block

        for key in to_exclude:
            del module[key]
        module["stake"] = round(from_rems(module["stake"]), 2)  # type: ignore
        if module.get("balance") is not None:
            module["balance"] = from_rems(module["balance"])  # type: ignore
        else:
            # user should not see None values
            del module["balance"]
        transformed_modules.append(module)

    return transformed_modules


def print_module_info(
    client: TorusClient,
    agents: list[AgentInfoWithOptionalBalance],
    console: Console,
    title: str | None = None,
) -> None:
    """
    Prints information about a module.
    """
    if not agents:
        return

    # Get the current block number, we will need this to caluclate immunity period
    block = client.get_block()
    if block:
        last_block = block["header"]["number"]
    else:
        raise ValueError("Could not get block info")

    # Get the immunity period on the netuid
    immunity_period = client.get_immunity_period()
    # tempo = client.get_tempo(netuid)

    # Transform the module dictionary to have immunity_period
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.DOUBLE_EDGE,
        title=title,
        caption_style="chartreuse3",
        title_style="bold magenta",
    )

    to_exclude = ["stake_from", "regblock"]
    tranformed_modules = transform_module_into(
        to_exclude, last_block, immunity_period, agents
    )

    sample_mod = tranformed_modules[0]
    for key in sample_mod.keys():
        # add columns
        table.add_column(key, style="white")

    total_stake = 0
    total_balance = 0

    for mod in tranformed_modules:
        total_stake += mod["stake"]
        if mod.get("balance") is not None:
            total_balance += mod["balance"]

        row: list[str] = []
        for val in mod.values():
            row.append(str(val))
        table.add_row(*row)

    table.caption = "total balance: " + f"{total_balance + total_stake}J"
    console.print(table)
    for _ in range(3):
        console.print()


def get_universal_password(ctx: CustomCtx) -> str:
    ctx.info("Please provide the universal password for all keys")
    universal_password = getpass()
    return universal_password


def tranform_network_params(params: dict[str, Any]):
    """Transform network params to be human readable."""
    params_ = params
    general_params = dict_from_nano(
        params_,
        [
            "min_weight_stake",
            "general_subnet_application_cost",
            "subnet_registration_cost",
            "proposal_cost",
            "max_proposal_reward_treasury_allocation",
        ],
    )

    return general_params
