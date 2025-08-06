import { createAsyncThunk } from "@reduxjs/toolkit";
import { axiosUserInstance } from "./axiosInstance";

interface AuthLoginPayload {
  email: string;
  password: string;
}

export const authUserLoginApi = createAsyncThunk(
  "auth/user/login",
  async (credentials: AuthLoginPayload, { rejectWithValue }) => {
    try {
      const response = await axiosUserInstance.post("/organization-user/login", credentials);
      return response.data;
    } catch (error) {
      return rejectWithValue(error);
    }
  }
);
