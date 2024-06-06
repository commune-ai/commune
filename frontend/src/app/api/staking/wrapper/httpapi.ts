import type {
    BaseQueryFn,
    FetchArgs,
    FetchBaseQueryError,
} from "@reduxjs/toolkit/query"
import { fetchBaseQuery } from "@reduxjs/toolkit/query"

const baseQuery = fetchBaseQuery({
    baseUrl: process.env.NEXT_PUBLIC_BACKEND_API,
    prepareHeaders: (headers) => {
        return headers
    },
})

const apiWrapper: BaseQueryFn<
    string | FetchArgs,
    unknown,
    FetchBaseQueryError
> = async (args, api, extraOptions) => {
    const result = await baseQuery(args, api, extraOptions)
    if (result.error && result.error.status === 401) {
        console.log('----------This is an error----', result.error)
    }
    return result
}

export default apiWrapper
