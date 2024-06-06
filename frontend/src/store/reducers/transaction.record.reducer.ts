import { SAVE_TRANSACTION_SUCCESS, SAVE_TRANSACTION_FAILED, SAVE_METAMASK_SUCCESS } from "../action/type";

const initialState = {
    records: {},
    loading: false,
    error: '',
    loginStatus: false,
    address: ''
}

export type SaveTransactionSuccessAction = {
    type: typeof SAVE_TRANSACTION_SUCCESS;
    payload: string;
}

export type SaveTransactionFailedAction = {
    type: typeof SAVE_TRANSACTION_FAILED;
    payload: string;
}

export type SaveMetamaskSuccessAction = {
    type: typeof SAVE_METAMASK_SUCCESS;
    payload: string
}

const transactionReducer = (state = initialState, action: SaveTransactionSuccessAction | SaveTransactionFailedAction | SaveMetamaskSuccessAction) => {

    const { type, payload } = action

    switch (type) {
        case SAVE_TRANSACTION_FAILED:
            return {
                ...state,
                error: payload
            }
        case SAVE_TRANSACTION_SUCCESS:
            return {
                ...state,
                records: payload
            }
        case SAVE_METAMASK_SUCCESS:
            console.log('--------------This is an error status-------', payload)
            return {
                ...state,
                address: payload,
                loginStatus: true
            }
        default:
            return state
    }

}

export default transactionReducer
