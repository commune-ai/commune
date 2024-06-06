import { DONE, LOADING, SAVE_METAMASK_FAILED, SAVE_METAMASK_SUCCESS } from "./type"

const API_URL = 'http://127.0.0.1:8000'

export const saveTransaction = async (payType: string, amount: number, destinationAddress: string, txHash: string) => {
    const body = JSON.stringify(
        {
            payType,
            amount,
            destinationAddress,
            txHash,
        })
    try {
        // const token = window.localStorage.getItem('token');
        await fetch(`${API_URL}/saveTransaction/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Accept: 'application/json',
            },
            body: body
        })
    }
    catch (e) {
        // dispatch({ type: SAVE_TRANSACTION_FAILED })
    }
}

interface Action {
    type: string;
    payload?: string;
}

export const saveMetaMaskAddress = (address: string) => async (dispatch: (action: Action) => void) => {

    const body = JSON.stringify(
        {
            address
        }
    )

    dispatch({ type: LOADING })

    dispatch({ type: SAVE_METAMASK_SUCCESS, payload: address })

    try { // const token = window.localStorage.getItem('token');

        const res = await fetch(`${API_URL}/saveMetamask/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Accept: 'application/json',
            },
            body: body
        })

        const data = await res.json()

        if (data.user_id) {
            dispatch({ type: SAVE_METAMASK_SUCCESS, payload: address })
        }

        dispatch({ type: DONE })

    }
    catch (e) {

        dispatch({ type: SAVE_METAMASK_FAILED })
        dispatch({ type: DONE })

    }

}
