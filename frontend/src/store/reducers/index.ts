import { configureStore } from "@reduxjs/toolkit"
import { setupListeners } from "@reduxjs/toolkit/query"
import { combineReducers } from 'redux'
import { statsApi } from "@/app/api/staking/modulelist"
import authReducer from './auth.reducer'
import transactionRecord from './transaction.record.reducer'

const rootReducer = combineReducers({
    stats: statsApi.reducer,
    transactionRecord: transactionRecord,
    authReducer: authReducer
});

export const store = configureStore({
    reducer: rootReducer,
})

setupListeners(store.dispatch)

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch
