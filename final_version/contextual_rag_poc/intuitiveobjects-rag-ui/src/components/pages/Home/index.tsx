import { useState, useEffect } from "react";
import { ChatWindow } from "@/components/organisms/ChatWindow/ChatWindow";
import { Header } from "@/components/organisms/Header/Header";
import { Sidebar } from "@/components/organisms/Sidebar/Sidebar";
import { useAppDispatch, useAppSelector } from "@/store/hooks";
import { setCurrentChat } from "@/store/slices/chatSlice";
import { useNavigate } from "react-router-dom";
import { createChatApi, getChatsApi } from "@/services/chatApis";

export const Home = () => {
  const navigate = useNavigate();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const dispatch = useAppDispatch();
  const isLoading = useAppSelector((state) => state.chat.isLoading);
  const history = useAppSelector((state) => state.chat.history);
  const [isLoadingSidebar] = useState(false);
  const [isLoadingMessages] = useState(false);

  const handleNewChat = async () => {
    const response = await dispatch(createChatApi("New Chat"));
    if (response.type === "chat/create/fulfilled") {
      navigate(`/chat/${response.payload.data.id}`);
    }
  };

  const fetchChats = async () => {
    const response = await dispatch(getChatsApi());
    if (response.type === "chat/getChats/fulfilled") {
      if (response.payload.data.length > 0) {
        navigate(`/chat/${response.payload.data[0].id}`);
      }
    }
  };

  useEffect(() => {
    fetchChats();
  }, []);

  return (
    <div className="h-screen flex flex-col">
      <Header onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)} />
      <Sidebar
        isOpen={isSidebarOpen}
        conversations={history.map((chat) => ({
          id: chat.id,
          title: chat.name,
          timestamp: chat.created_at,
        }))}
        onNewChat={handleNewChat}
        onSelectChat={(id) => {
          dispatch(setCurrentChat(id));
          navigate(`/chat/${id}`);
        }}
        onClose={() => setIsSidebarOpen(false)}
        isLoading={isLoadingSidebar}
      />

      <main className="flex-1 lg:ml-64">
        <ChatWindow onload={isLoadingMessages} isLoading={isLoading} />
      </main>
    </div>
  );
};
