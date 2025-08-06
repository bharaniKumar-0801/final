import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { Chat } from "@/services/chatApi";
import {
  getChatsApi,
  createChatApi,
  getChatMessagesApi,
  sendChatMessageApi,
  deleteChatApi,
  editChatApi,
} from "@/services/chatApis";

interface Message {
  id: string | null;
  content: string;
  role: "user" | "assistant";
  timestamp: string;
}

interface ChatState {
  chats: Chat[];
  currentChatId: string | null;
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  history: Chat[];
}

const initialState: ChatState = {
  chats: [],
  currentChatId: null,
  messages: [],
  isLoading: false,
  error: null,
  history: [],
};

const chatSlice = createSlice({
  name: "chat",
  initialState,
  reducers: {
    addChat: (state, action: PayloadAction<Chat>) => {
      state.chats.push(action.payload);
      state.currentChatId = action.payload.id;
    },
    setCurrentChat: (state, action: PayloadAction<string>) => {
      state.currentChatId = action.payload;
    },
    addMessage: (state, action: PayloadAction<Message>) => {
      state.messages.push(action.payload);
      if (action.payload.role === "assistant") {
        state.isLoading = false;
      }
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    resetMessages: (state) => {
      state.messages = [];
    },
  },
  extraReducers: (builder) => {
    builder.addCase(getChatsApi.fulfilled, (state, action) => {
      if (action.payload.data) {
        state.history = action.payload.data;
      }
    });
    builder.addCase(createChatApi.fulfilled, (state, action) => {
      if (action.payload.data) {
        state.history.push(action.payload.data);
      }
    });
    builder.addCase(getChatMessagesApi.fulfilled, (state, action) => {
      if (action.payload.data) {
        state.messages = action.payload.data;
      }
    });
    builder.addCase(sendChatMessageApi.fulfilled, (state, action) => {
      if (action.payload.data) {
        state.messages.pop();
        state.messages.push(action.payload.data.request_message);
        state.messages.push(action.payload.data.response_message);
      }
    });
    builder.addCase(deleteChatApi.fulfilled, (state, action) => {
      state.history = state.history.filter(
        (chat) => chat.id !== action.payload.data.id
      );
      state.currentChatId = null;
      state.messages = [];
    });
    builder.addCase(editChatApi.fulfilled, (state, action) => {
      state.history = state.history.map((chat) =>
        chat.id === action.payload.data.id ? action.payload.data : chat
      );
    });
  },
});

export const {
  addChat,
  setCurrentChat,
  addMessage,
  setLoading,
  resetMessages,
} = chatSlice.actions;
export default chatSlice.reducer;
