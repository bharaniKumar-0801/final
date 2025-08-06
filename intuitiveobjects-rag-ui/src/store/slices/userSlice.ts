import { createSlice, PayloadAction } from "@reduxjs/toolkit";

interface UserState {
  name: string;
  avatar?: string;
  isLoggedIn: boolean;
}

const initialState: UserState = {
  name: 'Guest User',
  avatar: undefined,
  isLoggedIn: false,
};

const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    setUser: (state, action: PayloadAction<{ name: string; avatar?: string }>) => {
      state.name = action.payload.name;
      state.avatar = action.payload.avatar;
      state.isLoggedIn = true;
    },
    logout: (state) => {
      state.name = 'Guest User';
      state.avatar = undefined;
      state.isLoggedIn = false;
    },
  },
});

export const { setUser, logout } = userSlice.actions;
export default userSlice.reducer; 