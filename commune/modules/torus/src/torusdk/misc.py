import re
from typing import Any, TypeVar

from torusdk._common import transform_stake_dmap
from torusdk.client import TorusClient
from torusdk.key import check_ss58_address
from torusdk.types.proposal import Emission
from torusdk.types.types import (
    Agent,
    AgentInfoWithOptionalBalance,
    GlobalGovernanceConfig,
    GlobalParams,
    MinFee,
    Ss58Address,
)

T = TypeVar("T")


def get_map_modules(
    client: TorusClient,
    include_balances: bool = False,
) -> dict[str, AgentInfoWithOptionalBalance]:
    """
    Gets all agents info on the network
    """
    request_dict: dict[Any, Any] = {
        "Torus0": [
            ("Agents", []),
            ("RegistrationBlock", []),
            ("StakedBy", []),
        ],
    }

    if include_balances:
        request_dict["System"] = [("Account", [])]
    bulk_query = client.query_batch_map(request_dict)
    (
        ss58_to_stakeby,
        ss58_to_agents,
        ss58_to_balances,
    ) = (
        bulk_query.get("StakedBy", {}),
        bulk_query.get("Agents", {}),
        bulk_query.get("Account", {}),
    )
    ss58_to_agents = {
        ss58: Agent.model_validate(agent)
        for ss58, agent in ss58_to_agents.items()
    }
    result_agents: dict[str, AgentInfoWithOptionalBalance] = {}
    ss58_to_stakeby = transform_stake_dmap(ss58_to_stakeby)
    for ss58 in ss58_to_agents.keys():
        key = check_ss58_address(ss58)
        regblock = ss58_to_agents[ss58].registration_block
        stake_from = ss58_to_stakeby.get(key, [])
        metadata = ss58_to_agents[ss58].metadata
        url = ss58_to_agents[ss58].url
        name = ss58_to_agents[ss58].name

        balance = None
        if include_balances and ss58_to_balances is not None:  # type: ignore
            balance_dict = ss58_to_balances.get(key, None)
            if balance_dict is not None:
                assert isinstance(balance_dict["data"], dict)
                balance = balance_dict["data"]["free"]
            else:
                balance = 0
        stake = sum(stake for _, stake in stake_from)

        agent: AgentInfoWithOptionalBalance = {
            "key": key,
            "name": name,
            "url": url,
            "stake_from": stake_from,
            "regblock": regblock,
            "balance": balance,
            "stake": stake,
            "metadata": metadata,
            "staking_fee": ss58_to_agents[ss58].fees.staking_fee,
            "weight_control_fee": ss58_to_agents[ss58].fees.weight_control_fee,
        }

        result_agents[key] = agent
    return result_agents


def to_snake_case(d: dict[str, T]) -> dict[str, T]:
    """
    Converts a dictionary with camelCase keys to snake_case keys
    """

    def snakerize(camel: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", camel).lower()

    snaked: dict[str, T] = {snakerize(k): v for k, v in d.items()}
    return snaked


def get_governance_config(c_client: TorusClient):
    governance_config = c_client.query_batch(
        {
            "Governance": [
                ("GlobalGovernanceConfig", []),
            ],
        }
    )["GlobalGovernanceConfig"]
    return GlobalGovernanceConfig.model_validate(governance_config)


def get_emission_params(c_client: TorusClient):
    query_all = c_client.query_batch(
        {
            "Emission0": [
                ("EmissionRecyclingPercentage", []),
                ("IncentivesRatio", []),
            ],
            "Governance": [
                ("TreasuryEmissionFee", []),
            ],
        }
    )
    raw_emission = {
        "recycling_percentage": query_all["EmissionRecyclingPercentage"],
        "treasury_percentage": query_all["TreasuryEmissionFee"],
        "incentives_ratio": query_all["IncentivesRatio"],
    }
    emission_params = Emission.model_validate(raw_emission)

    return emission_params


def get_fees(c_client: TorusClient):
    fees = c_client.query_batch(
        {
            "Torus0": [
                ("FeeConstraints", []),
            ],
        }
    )["FeeConstraints"]
    return MinFee.model_validate(fees)


def get_global_params(c_client: TorusClient):
    """
    Returns global parameters of the whole commune ecosystem
    """

    query_all = c_client.query_batch(
        {
            "Torus0": [
                ("MaxNameLength", []),
                ("MinNameLength", []),
                ("MaxAllowedAgents", []),
                ("DividendsParticipationWeight", []),
                ("FeeConstraints", []),
            ],
            "Emission0": [
                ("MaxAllowedWeights", []),
                ("MinStakePerWeight", []),
            ],
            "Governance": [
                ("GlobalGovernanceConfig", []),
            ],
        }
    )
    governance_config = GlobalGovernanceConfig.model_validate(
        query_all["GlobalGovernanceConfig"]
    )
    fees = MinFee.model_validate(query_all["FeeConstraints"])
    network_params = {
        "max_name_length": int(query_all["MaxNameLength"]),
        "min_name_length": int(query_all["MinNameLength"]),
        "max_allowed_agents": int(query_all["MaxAllowedAgents"]),
        "dividends_participation_weight": int(
            query_all["DividendsParticipationWeight"]
        ),
        "max_allowed_weights": int(query_all["MaxAllowedWeights"]),
        "min_weight_control_fee": fees.min_weight_control_fee,
        "min_staking_fee": fees.min_staking_fee,
        "proposal_cost": governance_config.proposal_cost,
        "min_stake_per_weight": query_all["MinStakePerWeight"],
        "proposal_expiration": governance_config.proposal_expiration,
        "agent_application_cost": governance_config.agent_application_cost,
        "agent_application_expiration": governance_config.agent_application_expiration,
        "proposal_reward_treasury_allocation": governance_config.proposal_reward_treasury_allocation,
        "max_proposal_reward_treasury_allocation": governance_config.max_proposal_reward_treasury_allocation,
        "proposal_reward_interval": governance_config.proposal_reward_interval,
    }
    global_params: GlobalParams = GlobalParams.model_validate(network_params)
    return global_params


def concat_to_local_keys(
    balance: dict[str, int], local_key_info: dict[str, Ss58Address]
) -> dict[str, int]:
    key2: dict[str, int] = {
        key_name: balance.get(key_address, 0)
        for key_name, key_address in local_key_info.items()
    }

    return key2


def local_keys_to_freebalance(
    c_client: TorusClient,
    local_keys: dict[str, Ss58Address],
) -> dict[str, int]:
    query_all = c_client.query_batch_map(
        {
            "System": [("Account", [])],
        }
    )
    balance_map = query_all["Account"]

    format_balances: dict[str, int] = {
        key: value["data"]["free"]
        for key, value in balance_map.items()
        if "data" in value and "free" in value["data"]
    }

    key2balance: dict[str, int] = concat_to_local_keys(
        format_balances, local_keys
    )

    return key2balance


def local_keys_to_stakedbalance(
    c_client: TorusClient,
    local_keys: dict[str, Ss58Address],
) -> dict[str, int]:
    staketo_map = c_client.query_map_staketo()

    format_stake: dict[str, int] = {
        key: sum(stake for _, stake in value)
        for key, value in staketo_map.items()
    }

    key2stake: dict[str, int] = concat_to_local_keys(format_stake, local_keys)

    return key2stake


def local_keys_to_stakedfrom_balance(
    c_client: TorusClient,
    local_keys: dict[str, Ss58Address],
) -> dict[str, int]:
    stakefrom_map = c_client.query_map_stakefrom()

    format_stake: dict[str, int] = {
        key: sum(stake for _, stake in value)
        for key, value in stakefrom_map.items()
    }

    key2stake: dict[str, int] = concat_to_local_keys(format_stake, local_keys)
    key2stake = {key: stake for key, stake in key2stake.items()}
    return key2stake


def local_keys_allbalance(
    c_client: TorusClient,
    local_keys: dict[str, Ss58Address],
) -> tuple[dict[str, int], dict[str, int]]:
    query_all = c_client.query_batch_map(
        {
            "System": [("Account", [])],
            "Torus0": [
                ("StakingTo", []),
            ],
        }
    )

    balance_map, staketo_map = (
        query_all["Account"],
        transform_stake_dmap(query_all.get("StakingTo", {})),
    )

    format_balances: dict[str, int] = {
        key: value["data"]["free"]
        for key, value in balance_map.items()
        if "data" in value and "free" in value["data"]
    }
    key2balance: dict[str, int] = concat_to_local_keys(
        format_balances, local_keys
    )
    format_stake: dict[str, int] = {
        key: sum(stake for _, stake in value)
        for key, value in staketo_map.items()
    }

    key2stake: dict[str, int] = concat_to_local_keys(format_stake, local_keys)

    key2balance = {
        k: v
        for k, v in sorted(
            key2balance.items(), key=lambda item: item[1], reverse=True
        )
    }

    key2stake = {
        k: v
        for k, v in sorted(
            key2stake.items(), key=lambda item: item[1], reverse=True
        )
    }

    return key2balance, key2stake


if __name__ == "__main__":
    from torusdk._common import get_node_url

    client = TorusClient(get_node_url(use_testnet=True))
    get_global_params(client)
