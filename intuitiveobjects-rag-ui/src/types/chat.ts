export interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: string;
}

export interface Conversation {
  id: string;
  title: string;
  timestamp: string;
}
