import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Loader } from "@/components/atoms/Loading/Loading";
import { useAppDispatch, useAppSelector } from "@/store/hooks";
import { useParams } from "react-router-dom";
import { Chat } from "@/services/chatApi";
import MoreHorizIcon from "@/components/atoms/Icon/MoreHorizIcon";
import { Menu, MenuButton, MenuItems, MenuItem } from "@headlessui/react";
import { TrashIcon } from "@/components/atoms/Icon/Trash";
import EditIcon from "@/components/atoms/Icon/EditIcon";
import { useState } from "react";
import { TextInput } from "@/components/atoms/TextInput";
import { deleteChatApi, editChatApi } from "@/services/chatApis";
import { toast } from "react-hot-toast";
import { confirmAction } from "@/utils/sweetAlert";
interface Conversation {
  id: string;
  title: string;
  timestamp: string;
}

interface SidebarProps {
  isOpen: boolean;
  conversations: Conversation[];
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  selectedId?: string;
  onClose?: () => void;
  isLoading: boolean;
}

export const Sidebar = ({
  isOpen,
  onNewChat,
  onSelectChat,
  onClose,
  isLoading,
}: SidebarProps) => {
  const history = useAppSelector((state) => state.chat.history);
  const { chatId } = useParams();

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 transition-opacity lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed top-0 left-0 h-full bg-sidebar-bg w-64 transform transition-transform duration-300 border-r border-chat-border ease-in-out ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        } lg:translate-x-0 z-20`}
      >
        <div className="flex flex-col h-full">
          <div className="p-4">
            <Button
              onClick={onNewChat}
              variant="secondary"
              className="w-full flex items-center justify-center gap-3"
            >
              <Icon name="plus" className="w-4 h-4" />
              New Chat
            </Button>
          </div>
          <div className="flex-1 overflow-y-auto">
            {isLoading ? (
              <Loader />
            ) : (
              history.map((conv) => (
                <ChatItem
                  key={conv.id}
                  conv={conv}
                  isActive={chatId === conv.id}
                  onSelectChat={onSelectChat}
                />
              ))
            )}
          </div>
        </div>
      </aside>
    </>
  );
};

interface ChatItemProps {
  conv: Chat;
  isActive: boolean;
  onSelectChat: (id: string) => void;
}

const ChatItem = ({ conv, isActive, onSelectChat }: ChatItemProps) => {
  const dispatch = useAppDispatch();

  const [isEditing, setIsEditing] = useState(false);
  const [editedName, setEditedName] = useState(conv.name);

  const onRenameChatBtnClick = () => {
    setIsEditing(true);
    setEditedName(conv.name);
  };

  const onDeleteBtnClick = () => {
    confirmAction(
      "Are you sure?",
      "You won't be able to revert this!",
      "Yes, delete it!"
    ).then(async (result) => {
      if (result.isConfirmed) {
        const res = await dispatch(deleteChatApi(conv.id));
        if (deleteChatApi.fulfilled.match(res)) {
          toast.success("Chat deleted");
        } else {
          toast.error("Failed to delete chat");
        }
      }
    });
  };

  const handleRenameChat = async () => {
    setIsEditing(false);

    if (editedName === conv.name || editedName.length === 0) {
      return;
    }

    const res = await dispatch(
      editChatApi({ chatId: conv.id, name: editedName })
    );
    if (editChatApi.fulfilled.match(res)) {
      toast.success("Chat renamed");
    }
  };

  return (
    <Button
      key={conv.id}
      onClick={() => onSelectChat(conv.id)}
      variant="ghost"
      className={`w-full text-left p-3 flex items-center gap-1 transition-colors duration-200 ${
        isActive
          ? "bg-[#2A2B32] text-white"
          : "text-gray-300 hover:bg-[#2A2B32] hover:text-white"
      }`}
    >
      <Icon name="chat" className={`w-4 h-4 ${isActive ? "text-white" : ""}`} />
      <div className="flex-1 min-w-0 flex items-center justify-between">
        <div className={`truncate px-1 ${isActive ? "text-white" : ""}`}>
          {isEditing ? (
            <TextInput
              value={editedName}
              onChange={(e) => setEditedName(e.target.value)}
              hideLabel
              className="!px-1 !py-0 rounded-md"
              onBlur={handleRenameChat}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  handleRenameChat();
                }
              }}
            />
          ) : (
            conv.name
          )}
        </div>
        {!isEditing ? (
          <Menu as="div" className="relative inline-block text-left">
            <MenuButton className="hover:bg-gray-500 rounded flex items-center justify-center">
              <MoreHorizIcon className="fill-white" />
            </MenuButton>
            <MenuItems className="absolute right-0 mt-2 w-32 origin-top-right bg-gray-600 rounded-md shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none z-50">
              <MenuItem>
                {() => (
                  <button
                    onClick={onDeleteBtnClick}
                    className={`text-white w-full text-left px-4 py-2 text-sm hover:bg-chat-bg flex items-center justify-between gap-2`}
                  >
                    Delete
                    <TrashIcon size={20} className="fill-white" />
                  </button>
                )}
              </MenuItem>
              <MenuItem>
                {() => (
                  <button
                    onClick={onRenameChatBtnClick}
                    className={`text-white w-full text-left px-4 py-2 text-sm hover:bg-chat-bg flex items-center justify-between gap-2`}
                  >
                    Rename
                    <EditIcon size={20} className="fill-white" />
                  </button>
                )}
              </MenuItem>
            </MenuItems>
          </Menu>
        ) : null}
      </div>
    </Button>
  );
};
