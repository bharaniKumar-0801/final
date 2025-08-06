import { MessageList } from "@/components/molecules/MessageList/MessageList";
import { ChatInput } from "@/components/molecules/ChatInput/ChatInput";
import { EmptyState } from "@/components/molecules/EmptyState/EmptyState";
import { Container } from "@/components/atoms/Container/Container";
import { useEffect, useRef } from "react";
import { Loader } from "@/components/atoms/Loading/Loading";
import { useAppDispatch, useAppSelector } from "@/store/hooks";
import { useParams } from "react-router-dom";
import { getChatMessagesApi } from "@/services/chatApis";

interface ChatWindowProps {
  isLoading?: boolean;
  onload?: boolean;
}

export const ChatWindow = ({ isLoading, onload }: ChatWindowProps) => {
  const messages = useAppSelector((state) => state.chat.messages);
  const history = useAppSelector((state) => state.chat.history);

  const dispatch = useAppDispatch();
  const { chatId } = useParams();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (chatId) {
      dispatch(getChatMessagesApi(chatId));
    }
  }, [chatId]);

  useEffect(() => {
    if (messages.length > 0) {
      scrollToBottom();
    }
  }, [messages, isLoading]);

  return (
    <div className="flex flex-col h-full bg-chat-bg">
      <div className="relative flex-1">
        <div className="absolute inset-0 overflow-y-auto">
          {onload ? (
            <Loader className="h-screen" />
          ) : !isLoading && messages.length === 0 ? (
            <EmptyState
              title="How can I help you today?"
              description="Ask me anything!"
            />
          ) : (
            <Container>
              <MessageList
                messages={messages.map((message) => ({
                  ...message,
                  id: message.id || "",
                }))}
                isLoading={isLoading}
                messagesEndRef={messagesEndRef}
              />
            </Container>
          )}
        </div>
      </div>
      {chatId && (
        <div className="sticky bottom-0 border-t border-chat-border bg-chat-bg">
          <Container className="p-4">
            <ChatInput disabled={history.length === 0} />
          </Container>
        </div>
      )}
    </div>
  );
};
