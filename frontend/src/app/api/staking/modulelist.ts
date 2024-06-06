import { createApi } from "@reduxjs/toolkit/query/react"
import {
    IBalanceType,
    IStats,
    InterfacePagination,
    InterfacePaginationSubnet,
    SubnetInterface,
    ValidatorType,
} from "@/types"
import apiWrapper from "./wrapper/httpapi"
export const statsApi = createApi({
    reducerPath: "statsApi",
    baseQuery: apiWrapper,
    tagTypes: [
        "ValidatorsList",
        "CommuneStats",
        "SingleValidator",
        "SubnetsList",
        "SingleSubnet",
    ],
    endpoints: (builder) => ({
        getValidators: builder.query<ValidatorType[], void>({
            query: () => "/validators/",
            providesTags: ["ValidatorsList"],
            transformResponse: (response: InterfacePagination<ValidatorType[]>) => {
                const validatedResponse: ValidatorType[] = response.validators.map(
                    (validator) => {
                        validator.isVerified = validator.expire_at === -1 || (validator.expire_at || 0) > Date.now() / 1000
                        return validator
                    },
                )
                return validatedResponse.toSorted((a,) =>
                    a.key === process.env.NEXT_PUBLIC_COMSTAT_VALIDATOR ? -1 : 1,

                )
            },
        }),
        getValidatorsById: builder.query<
            ValidatorType,
            { key: string; wallet: string, subnet_id?: number }
        >({
            query: ({ key, wallet, subnet_id = 0 }) => {
                let url = `/validators/${key}?subnet_id=${subnet_id}`
                if (wallet && wallet !== undefined) {
                    url += `&wallet=${wallet}`
                }
                return url
            },
            providesTags: (_, __, { key }) => [{ type: "SingleValidator", id: key }],
            transformResponse: (response: ValidatorType) => {
                const validatedResponse: ValidatorType = {
                    ...response,
                    isVerified: response.expire_at === -1 || (response.expire_at || 0) > Date.now() / 1000,
                }
                console.log(validatedResponse)
                // validatedResponse.stake_from = validatedResponse?.stake_from?.sort(
                //     (a, b) => b[1] - a[1],
                // )
                return validatedResponse
            },
        }),
        getSubnets: builder.query<SubnetInterface[], void>({
            query: () => "/subnets/",
            providesTags: ["SubnetsList"],
            transformResponse: (
                response: InterfacePaginationSubnet<SubnetInterface[]>,
            ) => {
                // const validatedResponse: ValidatorType[] = response.validators.map(
                //   (validator) => {
                //     if (verifiedValidators.some((v) => v.key === validator.key)) {
                //       validator.isVerified = true
                //     } else {
                //       validator.isVerified = false
                //     }
                //     return validator
                //   },
                // )
                // return validatedResponse.toSorted((a, b) =>
                //   a.key === process.env.NEXT_PUBLIC_COMSTAT_VALIDATOR ? -1 : 1,
                // )
                return response.subnets
            },
        }),
        getSubnetById: builder.query<ValidatorType[], string>({
            query: (id) => `/validators/?subnet_id=${id}`,
            providesTags: (_, __, id) => [{ type: "SingleSubnet", id: id }],
            transformResponse: (response: InterfacePagination<ValidatorType[]>) => {
                const validatedResponse: ValidatorType[] = response.validators.map(
                    (validator) => {
                        validator.isVerified = validator.expire_at === -1 || (validator.expire_at || 0) > Date.now() / 1000
                        return validator
                    },
                )
                return validatedResponse.toSorted((a,) =>
                    a.key === process.env.NEXT_PUBLIC_COMSTAT_VALIDATOR ? -1 : 1,
                )
            },
        }),
        getTotalStats: builder.query<IStats, void>({
            query: () => "/stats/",
            providesTags: ["CommuneStats"],
            transformResponse: (response: { stats: IStats }) => {
                return response.stats
            },
        }),
        getBalance: builder.query<IBalanceType, { wallet: string }>({
            query: ({ wallet }) => `/balance/?wallet=${wallet}`,
            providesTags: ["SingleValidator"],
            transformResponse: (response: IBalanceType) => {
                return response
            },
        }),
        searchBalance: builder.mutation<IBalanceType, { wallet: string }>({
            query: ({ wallet }) => `/balance/?wallet=${wallet}`,
            transformResponse: (response: IBalanceType) => {
                return response
            },
        }),
    }),
})

export const {
    useGetValidatorsQuery,
    useGetBalanceQuery,
    useGetTotalStatsQuery,
    useSearchBalanceMutation,
    useGetValidatorsByIdQuery,
} = statsApi
