export type ValidatorType = {
    subnet_id: number,
    name: string
    address: string
    emission: number
    incentive: number
    dividends: number
    regblock: number
    last_update: number
    balance: number
    weight: number
    stake: number
    total_stakers: number
    delegation_fee: number
    type: string
    key: string
    apy: number
    wallet_staked?: number
    isVerified?: boolean
}

export interface InterfacePagination<Data> {
    total: number
    page: number
    limit: number
    validators: Data
}

export type IAddStaking = {
    validator: string
    amount: string
    callback?: () => void
}

export type ITransfer = {
    to: string
    amount: string
    callback?: () => void
}
export type ITransferStaking = {
    validatorFrom: string
    amount: string
    validatorTo: string
    callback?: () => void
}
export interface IStats {
    circulating_supply: number
    total_stake: number
    total_subnets: number
    total_validators: number
    total_miners: number
    total_modules: number
    price: number
    marketcap: number
    daily_emission: number
    total_stakers: number
    avg_apy: number
}
export interface IBalanceType {
    balance: number
    staked: number
    stakes: { amount: number; validator: ValidatorType }[]
}
