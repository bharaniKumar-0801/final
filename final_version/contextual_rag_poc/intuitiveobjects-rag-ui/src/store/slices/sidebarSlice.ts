import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { Chat } from "@/services/chatApi";

const initialState: Chat[] = [];

const sidebarSlice = createSlice({
	name: 'sidebar',
	initialState,
	reducers: {
        setChatHistory: (_, action: PayloadAction<Chat[]>) => {
			return action.payload;
		},
	},
});

export const { setChatHistory } = sidebarSlice.actions;
export default sidebarSlice.reducer; 