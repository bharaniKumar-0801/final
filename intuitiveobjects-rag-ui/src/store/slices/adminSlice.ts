import { createSlice } from "@reduxjs/toolkit";
import {
	orgCreateCategoryApi,
	orgCreateUserApi,
	orgDeleteCategoryApi,
	orgGetCategoriesApi,
	orgGetUsersApi,
	orgDeleteUserApi,
	orgGetFilesApi,
	orgDeleteFileApi,
	orgIngestDocumentApi,
	settingGetAppConfigApi,
	settingUpdateAppConfigApi,
	settingGetAppCurrentConfigApi,
} from "@/services/adminApi";

interface AdminState {
	categories: any[];
	users: any[];
	files: any[];
	loading: boolean;
	error: string | null;
	appConfig: any;
	currentAppConfig: any;
}

const initialState: AdminState = {
	categories: [],
	users: [],
	files: [],
	loading: false,
	error: null,
	appConfig: null,
	currentAppConfig: null,
};

const adminSlice = createSlice({
	name: "admin",
	initialState,
	reducers: {},
	extraReducers: (builder) => {
		builder.addCase(orgGetCategoriesApi.fulfilled, (state, action) => {
			state.categories = action.payload.data;
		});
		builder.addCase(orgCreateCategoryApi.fulfilled, (state, action) => {
			state.categories.push(action.payload.data);
		});
		builder.addCase(orgDeleteCategoryApi.fulfilled, (state, action) => {
			state.categories = state.categories.filter((category) => category.id !== action.payload.data.id);
		});
		builder.addCase(orgCreateUserApi.fulfilled, (state, action) => {
			if (action.payload.organization_user) {
				state.users.push(action.payload.organization_user);
			}
		});
		builder.addCase(orgGetUsersApi.fulfilled, (state, action) => {
			if (action.payload.organization_users) {
				state.users = action.payload.organization_users;
			}
		});
		builder.addCase(orgDeleteUserApi.fulfilled, (state, action) => {
			state.users = state.users.filter((user) => user.id !== action.payload.data.id);
		});
		builder.addCase(orgIngestDocumentApi.fulfilled, (state, action) => {
			if (action.payload.data) {
				state.files.push(action.payload.data);
			}
		});
		builder.addCase(orgGetFilesApi.fulfilled, (state, action) => {
			if (action.payload.data) {
				state.files = action.payload.data;
			}
		});
		builder.addCase(orgDeleteFileApi.fulfilled, (state, action) => {
			if (action.payload.data) {
				state.files = state.files.filter((file) => file.id !== action.payload.data.id);
			}
		});
		builder.addCase(settingGetAppConfigApi.fulfilled, (state, action) => {
			if (action.payload.data) {
				state.appConfig = action.payload.data;
			}
		});
		builder.addCase(settingUpdateAppConfigApi.fulfilled, (state, action) => {
			if (action.payload.data) {
				state.currentAppConfig = action.payload.data;
			}
		});
		builder.addCase(settingGetAppCurrentConfigApi.fulfilled, (state, action) => {
			if (action.payload.data) {
				state.currentAppConfig = action.payload.data?.[0];
			}
		});
	},
});

export default adminSlice.reducer;
