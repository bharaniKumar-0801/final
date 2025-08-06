import api from "./api";
import { Message } from "../types/chat";

interface ChatMessage {
  content: string;
  role: "user" | "assistant";
  timestamp: string;
}

export interface Chat {
  id: string;
  name: string;
  user_id: string;
  created_at: string;
}

export const chatApi = {
	sendMessage: (chatId: string, userId: string, content: string) =>
		api.post<ChatMessage>(`/chat/${chatId}/guest/${userId}/message`, {
			content,
		}).then(response => response.data),
	createChat: (name: string, userId: string) =>
		api.post<Chat>(`/chat/guest`, {
			name,
			user_id: userId
		}).then(response => response.data),
	getGuestChats: (userId: string) =>
		api.get<{ success: boolean; message: string; data: Chat[] }>(
			`/chat/guest/${userId}`
		).then(response => response.data || []),
	getChatMessages: (chatId: string, userId: string) =>
		api.get<{ success: boolean; message: string; data: Message[] }>(
			`/chat/${chatId}/guest/${userId}/messages`
		).then(response => response.data || []),
}; 