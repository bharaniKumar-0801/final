import { configureStore, combineReducers } from "@reduxjs/toolkit";
import { persistReducer, persistStore } from "redux-persist";
import storage from "redux-persist/lib/storage";
import chatReducer from "@/store/slices/chatSlice";
import userReducer from "@/store/slices/userSlice";
import sidebarReducer from "@/store/slices/sidebarSlice";
import authReducer from "@/store/slices/authSlice";
import adminReducer from "@/store/slices/adminSlice";

const persistConfig = {
  key: "root",
  storage,
};

const rootReducer = combineReducers({
  chat: chatReducer,
  user: userReducer,
  sidebar: sidebarReducer,
  auth: authReducer,
  admin: adminReducer,
});

const persistedReducer = persistReducer(persistConfig, rootReducer);

export const store = configureStore({
  reducer: persistedReducer,
});

export const persistor = persistStore(store);

const makeStore = { reduxStore: store, persistor };

export default makeStore;

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
