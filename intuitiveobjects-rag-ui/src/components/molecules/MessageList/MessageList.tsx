import { ChatMessage } from "@/components/molecules/ChatMessage/ChatMessage";
import { Loading } from "@/components/atoms/Loading/Loading";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: string;
}

interface MessageListProps {
  messages: Message[];
  isLoading?: boolean;
  messagesEndRef?: React.RefObject<HTMLDivElement>;
}

export const MessageList = ({
  messages,
  isLoading,
  messagesEndRef,
}: MessageListProps) => {
  return (
    <div>
      {messages.map((message) => (
        <ChatMessage
          key={message.id}
          content={message.content}
          role={message.role}
          timestamp={message.timestamp}
        />
      ))}
      {isLoading && <Loading />}
      {messagesEndRef && <div ref={messagesEndRef} />}
    </div>
  );
};
