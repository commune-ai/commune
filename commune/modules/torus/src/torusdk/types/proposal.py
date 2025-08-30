from typing import Any, Literal, Union, cast

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from torusdk.types.types import GlobalParams, Rem, Ss58Address, instantiate_rem


class ProposalOpen(BaseModel):
    # votes_for: list[Ss58Address]
    # votes_against: list[Ss58Address]
    status: Literal["Open"] = "Open"
    stake_for: Rem
    stake_against: Rem

    @field_serializer("stake_for", "stake_against")
    def from_rem(self, rem_value: Rem) -> int:
        return rem_value.value

    to_rem = field_validator("stake_for", "stake_against", mode="before")(
        instantiate_rem
    )

    class Config:
        arbitrary_types_allowed = True


class ProposalRefused(BaseModel):
    status: Literal["Refused"] = "Refused"
    block: int
    stake_for: Rem
    stake_against: Rem

    @field_serializer(
        "stake_for",
        "stake_against",
    )
    def from_rem(self, rem_value: Rem) -> int:
        return rem_value.value

    to_rem = field_validator("stake_for", "stake_against", mode="before")(
        instantiate_rem
    )

    class Config:
        arbitrary_types_allowed = True


class ProposalAccepted(BaseModel):
    status: Literal["Accepted"] = "Accepted"
    block: int
    stake_for: Rem
    stake_against: Rem

    @field_serializer("stake_for", "stake_against")
    def from_rem(self, rem_value: Rem) -> int:
        return rem_value.value

    to_rem = field_validator("stake_for", "stake_against", mode="before")(
        instantiate_rem
    )

    class Config:
        arbitrary_types_allowed = True


class ProposalExpired(BaseModel):
    status: Literal["Expired"] = "Expired"


class TransferDaoTreasury(BaseModel):
    account: Ss58Address
    amount: Rem = Field(..., init=True)

    to_rem = field_validator("amount", mode="before")(instantiate_rem)

    @field_serializer(
        "amount",
    )
    def from_rem(self, rem_value: Rem) -> int:
        return rem_value.value

    class Config:
        arbitrary_types_allowed = True


class OptionalEmission(BaseModel):
    recycling_percentage: int | None = Field(..., ge=0, le=100)
    treasury_percentage: int | None = Field(..., ge=0, le=100)


class Emission(BaseModel):
    recycling_percentage: int = Field(..., ge=0, le=100)
    treasury_percentage: int = Field(..., ge=0, le=100)
    incentives_ratio: int = Field(..., ge=0, le=100)


class GlobalCustom(BaseModel):
    pass


class ProposalData(BaseModel):
    global_params: GlobalParams | None = Field(None, alias="GlobalParams")
    emission: Emission | None = Field(None, alias="Emission")
    transfer_dao_treasury: TransferDaoTreasury | None = Field(
        None, alias="TransferDaoTreasury"
    )
    custom: GlobalCustom | None = Field(None, alias="GlobalCustom")


class Proposal(BaseModel):
    proposal_id: int = Field(..., alias="id")
    proposer: Ss58Address
    expiration_block: int
    status: ProposalAccepted | ProposalRefused | ProposalOpen | ProposalExpired
    metadata: str
    proposal_cost: Rem = Field(...)
    creation_block: int
    data: Union[GlobalParams, Emission, TransferDaoTreasury, GlobalCustom]

    class Config:
        arbitrary_types_allowed = True

    @field_serializer(
        "proposal_cost",
    )
    def from_rem(self, rem_value: Rem) -> int:
        return rem_value.value

    # TODO: find a better way to do this and remove this cursed thing
    @model_validator(mode="before")
    @classmethod
    def unwrap_data(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        data = cast(dict[Any, Any], data)
        if not data.get("data"):
            raise ValueError("Data must contain a 'data' key")
        if not isinstance(data["data"], dict):
            value = data["data"]
            data["data"] = {"data": value}
            return data
        if len(data.get("data")) != 1:
            raise ValueError("Data must contain only one key")
        data["data"] = [*data["data"].values()][0]
        return data

    @model_validator(mode="before")
    @classmethod
    def fix_status(cls, data: Any) -> Any:
        return extract_value(data, "status")

    to_rem = field_validator("proposal_cost", mode="before")(instantiate_rem)


def extract_value(data: Any, key_to_extract: str):
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary")
    data = cast(dict[Any, Any], data)
    if not data.get(key_to_extract):
        raise ValueError("Data must contain a 'data' key")
    if not isinstance(data[key_to_extract], dict):
        value = data[key_to_extract]
        data[key_to_extract] = {key_to_extract: value}
        return data
    if len(data.get(key_to_extract)) != 1:
        raise ValueError("Data must contain only one key")
    data[key_to_extract] = [*data[key_to_extract].values()][0]
    return data
