import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useAppDispatch } from "@/store/hooks";
import { Menu, MenuButton, MenuItem, MenuItems } from "@headlessui/react";
import { userLogout } from "@/store/slices/authSlice";
import { useNavigate } from "react-router-dom";
import PersonIcon from "@/components/atoms/Icon/PersonIcon";

interface HeaderProps {
  onToggleSidebar: () => void;
}

export const Header = ({ onToggleSidebar }: HeaderProps) => {
  const dispatch = useAppDispatch();
  const navigate = useNavigate();

  const handleLogout = () => {
    dispatch(userLogout());
    navigate("/login");
  };

  return (
    <header className="bg-chat-bg border-b border-chat-border h-14 flex items-center px-4">
      <Button
        onClick={onToggleSidebar}
        variant="ghost"
        className="lg:hidden"
        aria-label="Toggle Sidebar"
      >
        <Icon name="menu" className="w-6 h-6" />
      </Button>
      <div className="flex items-center justify-between w-full">
        <h1 className="text-white text-xl font-semibold">New Chat</h1>
        <div className="flex items-center gap-4">
          <div className="relative">
            <Menu as="div" className="relative">
              <MenuButton className="text-white">
                <Button variant="ghost" className="!rounded-full !p-2.5">
                  <PersonIcon className="fill-white" size={24} />
                </Button>
              </MenuButton>
              <MenuItems className="absolute right-0 mt-2 w-48 origin-top-right rounded-md bg-sidebar-bg shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                <MenuItem>
                  <button
                    className="text-white w-full text-left px-4 py-2 text-sm hover:bg-chat-bg rounded-md"
                    onClick={handleLogout}
                  >
                    Logout
                  </button>
                </MenuItem>
              </MenuItems>
            </Menu>
          </div>
        </div>
      </div>
    </header>
  );
};
