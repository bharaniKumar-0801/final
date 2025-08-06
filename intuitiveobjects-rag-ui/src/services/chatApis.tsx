import { createAsyncThunk } from "@reduxjs/toolkit";
import { axiosUserInstance } from "./axiosInstance";

export const createChatApi = createAsyncThunk(
  "chat/create",
  async (name: string, { rejectWithValue }) => {
    try {
      const response = await axiosUserInstance.post("/chat/user", { name });
      return response.data;
    } catch (error) {
      return rejectWithValue(error);
    }
  }
);

interface SendChatMessagePayload {
  chatId: string;
  content: string;
}

export const sendChatMessageApi = createAsyncThunk(
  "chat/sendMessage",
  async (value: SendChatMessagePayload, { rejectWithValue }) => {
    try {
      const response = await axiosUserInstance.post(
        `/chat/${value.chatId}/user/message`,
        { content: value.content }
      );
      return response.data;
    } catch (error) {
      return rejectWithValue(error);
    }
  }
);

export const getChatMessagesApi = createAsyncThunk(
  "chat/getMessages",
  async (chatId: string, { rejectWithValue }) => {
    try {
      const response = await axiosUserInstance.get(
        `/chat/${chatId}/user/messages`
      );

      return response.data;
    } catch (error) {
      return rejectWithValue(error);
    }
  }
);

export const getChatsApi = createAsyncThunk(
  "chat/getChats",
  async (_, { rejectWithValue }) => {
    try {
      const response = await axiosUserInstance.get("/chat/user");

      return response.data;
    } catch (error) {
      return rejectWithValue(error);
    }
  }
);

export const deleteChatApi = createAsyncThunk(
  "chat/delete",
  async (chatId: string, { rejectWithValue }) => {
    try {
      const response = await axiosUserInstance.delete(`/chat/${chatId}/user`);

      return response.data;
    } catch (error) {
      return rejectWithValue(error);
    }
  }
);

interface EditChatPayload {
  chatId: string;
  name: string;
}

export const editChatApi = createAsyncThunk(
  "chat/edit",
  async (payload: EditChatPayload, { rejectWithValue }) => {
    try {
      const response = await axiosUserInstance.put(
        `/chat/${payload.chatId}/user`,
        { name: payload.name }
      );

      return response.data;
    } catch (error) {
      return rejectWithValue(error);
    }
  }
);
