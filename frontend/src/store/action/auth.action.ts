import axios from "axios";
import { NotificationManager } from "react-notifications";
import { AnyAction } from "redux";
import { ThunkAction } from "redux-thunk";
import {
  PUBKEY,
  LOGOUT,
  LOGIN_USERS_REQUEST,
  LOGIN_USERS_SUCCESS,
  LOGIN_USERS_FAILURE,
  UPDATE_USERS_FAILURE,
  UPDATE_USERS_REQUEST,
  UPDATE_USERS_SUCCESS,
  GET_USERS_FAILURE,
  GET_USERS_REQUEST,
  GET_USERS_SUCCESS,
  POST_USERS_FAILURE,
  POST_USERS_REQUEST,
  POST_USERS_SUCCESS,
} from "./type";
import { RootState } from "../reducers"; // Adjust the import based on your project structure


interface User {
  id: string;
  name: string;
  email: string;
  // Add other user fields as necessary
}

interface LoginPayload {
  email: string;
  password: string;
}

interface CreateUserPayload {
  name: string;
  email: string;
  password: string;
  // Add other fields as necessary
}

interface UpdateUserPayload {
  userId: string;
  // Add other fields as necessary
}


export const getUser = (payload: string) => async (dispatch: any) => {
  dispatch({ type: GET_USERS_REQUEST });
  try {
    const res = await axios.get<{ user: User }>(`${process.env.NEXT_PUBLIC_BACKEND_URL}auth/${payload}`);
    dispatch({ type: GET_USERS_SUCCESS, payload: res.data.user });
  } catch (error) {
    dispatch({ type: GET_USERS_FAILURE, payload: error });
  }
};

export const createUser = (payload: CreateUserPayload) => async (dispatch: any) => {
  dispatch({ type: POST_USERS_REQUEST });
  try {
    const res = await axios.post<{ user: User }>(`${process.env.NEXT_PUBLIC_BACKEND_URL}api/auth/register`, payload);
    NotificationManager.success('User created successfully', 'Success');
    dispatch({ type: POST_USERS_SUCCESS, payload: res.data });
  } catch (error: any) {
    NotificationManager.error(error.response.data.msg, 'Error');
    dispatch({ type: POST_USERS_FAILURE, payload: error });
  }
};

export const login = (payload: LoginPayload) => async (dispatch: any) => {
  dispatch({ type: LOGIN_USERS_REQUEST });
  try {
    const res = await axios.post<{ token: string; user: User }>(`${process.env.NEXT_PUBLIC_BACKEND_URL}api/auth/signin`, payload);
    axios.defaults.headers.common["Authorization"] = res.data.token;
    localStorage.setItem("token", res.data.token);
    NotificationManager.success('User logged in', 'Success');
    dispatch({ type: LOGIN_USERS_SUCCESS, payload: res.data });
  } catch (error: any) {
    NotificationManager.error(error.response.data.msg, 'Error');
    dispatch({ type: LOGIN_USERS_FAILURE, payload: error });
  }
};

export const logOut = () => async (dispatch: any) => {
  dispatch({ type: LOGOUT });
};

export const tokenLogin = () => async (dispatch: any) => {
  dispatch({ type: LOGIN_USERS_REQUEST });
  try {
    const res = await axios.get<{ token: string; user: User }>(`${process.env.NEXT_PUBLIC_BACKEND_URL}api/auth/auth/tokenLogin`);
    axios.defaults.headers.common["Authorization"] = res.data.token;
    localStorage.setItem("token", res.data.token);
    dispatch({ type: LOGIN_USERS_SUCCESS, payload: res.data });
  } catch (error) {
    dispatch({ type: LOGIN_USERS_FAILURE, payload: error });
  }
};

export const Pubkey = (payload: string) => async (dispatch: any) => {
  dispatch({ type: PUBKEY, payload });
};

export const UpdateUser = (payload: UpdateUserPayload) => async (dispatch: any) => {
  dispatch({ type: UPDATE_USERS_REQUEST });
  try {
    const res = await axios.put<{ user: User }>(`${process.env.NEXT_PUBLIC_BACKEND_URL}api/auth/${payload.userId}`);
    dispatch({ type: UPDATE_USERS_SUCCESS, payload: res.data });
  } catch (error) {
    dispatch({ type: UPDATE_USERS_FAILURE, payload: error });
  }
};
