import { useState, KeyboardEvent, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { TextArea } from "@/components/atoms/TextArea/TextArea";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useAppDispatch } from "@/store/hooks";
import { addMessage, setLoading } from "@/store/slices/chatSlice";
import { useParams } from "react-router-dom";
import { sendChatMessageApi } from "@/services/chatApis";

interface ChatInputProps {
  disabled?: boolean;
}

export const ChatInput = ({ disabled }: ChatInputProps) => {
  const [message, setMessage] = useState("");
  const [isWaiting, setIsWaiting] = useState(false); // State to track if waiting for a response
  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  const dispatch = useAppDispatch();
  const { chatId } = useParams();

  const sendMessage = useCallback(async () => {
    if (message.trim() && !disabled && chatId) {
      setIsWaiting(true); // Set waiting state to true
      dispatch(setLoading(true));
      dispatch(
        addMessage({
          id: null,
          content: message,
          role: "user",
          timestamp: new Date().toISOString(),
        })
      );
      setMessage(""); // Clear the input field after sending
      await dispatch(sendChatMessageApi({ chatId, content: message.trim() }));
      dispatch(setLoading(false));
      // setMessage("");
    }
  }, [message, disabled, chatId, dispatch]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await sendMessage();
  };

  const handleKeyDown = async (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      await sendMessage();
    }
  };

  useEffect(() => {
    const textarea = textAreaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
    }
  }, [message]);

  return (
    <form
      onSubmit={handleSubmit}
      className="relative bg-chat-bg rounded-lg shadow-lg"
    >
      <TextArea
        ref={textAreaRef}
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={handleKeyDown}
        className="w-full pr-12"
        placeholder="Send a message..."
        disabled={disabled || isWaiting} // Disable input when waiting for a response
      />
      <Button
        type="submit"
        disabled={!message.trim() || disabled || isWaiting} // Disable button if input is empty, disabled, or waiting for a response
        variant="ghost"
        className="absolute right-2 bottom-[10px] p-1"
      >
        <Icon name="send" className="w-5 h-5 md:w-6 md:h-6" />
      </Button>
    </form>
  );
};
