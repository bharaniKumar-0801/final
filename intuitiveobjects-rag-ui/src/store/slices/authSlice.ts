import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { orgAdminLoginApi } from "@/services/adminApi";
import { authUserLoginApi } from "@/services/authApi";
interface User {
	token: string | null;
	isAuthenticated: boolean;
	user: any;
}

interface Admin {
	token: string | null;
	isAuthenticated: boolean;
	user: any;
}

interface AuthState {
	user: User | null;
	admin: Admin | null;
}

const initialState: AuthState = {
	user: null,
	admin: null,
};

export const authSlice = createSlice({
	name: "auth",
	initialState,
	reducers: {
		adminLogout: (state) => {
			state.admin = null;
		},
		userLogout: (state) => {
			state.user = null;
		},
	},
	extraReducers: (builder) => {
		builder.addCase(orgAdminLoginApi.fulfilled, (state, action) => {
			state.admin = {
				token: action.payload.data.token,
				isAuthenticated: true,
				user: action.payload.data.user,
			};
		});
		builder.addCase(authUserLoginApi.fulfilled, (state, action) => {
			if (action.payload.organization_user) {
				state.user = {
					token: action.payload.organization_user.token,
					isAuthenticated: true,
					user: action.payload.organization_user,
				};
			}
		});
		builder.addCase(logoutAdmin.fulfilled, (state) => {
			state.admin = {
				isAuthenticated: false,
				user: null,
				token: null,
			};
		});
	},
});

export const { adminLogout, userLogout } = authSlice.actions;

export const logoutAdmin = createAsyncThunk("auth/logoutAdmin", async (_, { rejectWithValue }) => {
	try {
		// Remove token from localStorage or wherever you store it
		localStorage.removeItem("admin_token");
		return true;
	} catch (error) {
		return rejectWithValue(error);
	}
});

export default authSlice.reducer;
