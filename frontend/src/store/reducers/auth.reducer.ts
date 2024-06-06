import {
    GET_USERS_FAILURE,
    GET_USERS_REQUEST,
    GET_USERS_SUCCESS,
    POST_USERS_FAILURE,
    POST_USERS_REQUEST,
    POST_USERS_SUCCESS,
    UPDATE_USERS_FAILURE,
    UPDATE_USERS_REQUEST,
    UPDATE_USERS_SUCCESS,
    LOGIN_USERS_REQUEST,
    LOGIN_USERS_SUCCESS,
    LOGIN_USERS_FAILURE,
    LOGOUT,
    PUBKEY,
} from "../action/type";

const initialState = {
    // initialState for get all users
    isLoading: false,
    users: [],
    user: null,
    error: null,
    pubkey: "",

    // initialState for create a user
    isLoadingPost: false,
    successPost: null,
    errorPost: null,

    isLogged: false,
    // initialState for update a user
    isLoadingUpdate: false,
    successUpdate: null,
    errorUpdate: null,
};

const authReducer = (state = initialState, action: any) => {
    switch (action.type) {
        // all user gets reducers
        case GET_USERS_REQUEST:
            return {
                ...state,
                isLoading: true,
            };
        case GET_USERS_SUCCESS:
            return {
                isLogged: true,
                isLoading: false,
                user: action.payload,
                error: null,
            };
        case GET_USERS_FAILURE:
            return {
                isLoading: false,
                user: null,
                error: action.payload,
            };

        // single user create reducers
        case POST_USERS_REQUEST:
            return {
                ...state,
                isLoadingPost: true,
            };
        case POST_USERS_SUCCESS:
            return {
                ...state,
                isLoadingPost: false,
                successPost: action.payload,
                errorPost: null,
            };
        case POST_USERS_FAILURE:
            return {
                ...state,
                isLoadingPost: false,
                successPost: null,
                errorPost: action.payload,
            };

        case LOGIN_USERS_REQUEST:
            return {
                ...state,
                isLoadingPost: true,
            };
        case LOGIN_USERS_SUCCESS:
            return {
                ...state,
                isLogged: true,
                isLoadingPost: false,
                user: action.payload.user,
                errorPost: null,
            };
        case LOGIN_USERS_FAILURE:
            return {
                ...state,
                isLoadingPost: false,
                user: null,
                errorPost: action.payload,
            };
        case LOGOUT:
            return {
                ...state,
                isLogged: false,
                user: null,
            };

        // single user update reducers
        case UPDATE_USERS_REQUEST:
            return {
                ...state,
                isLoadingUpdate: true,
            };
        case UPDATE_USERS_SUCCESS:
            return {
                ...state,
                isLoadingUpdate: false,
                successUpdate: action.payload,
                errorUpdate: null,
            };
        case UPDATE_USERS_FAILURE:
            return {
                ...state,
                isLoadingUpdate: false,
                successUpdate: null,
                errorUpdate: action.payload,
            };
        case PUBKEY:
            return {
                ...state,
                pubkey: action.payload,
            };

        default:
            return state;
    }
};

export default authReducer;
