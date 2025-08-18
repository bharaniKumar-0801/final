import { Icon } from "@/components/atoms/Icon/Icon";

export const AdminSidebar = () => {
  return (
    <aside className="fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 bg-sidebar-bg border-r border-chat-border hidden lg:block">
      <nav className="p-4">
        <div className="space-y-2">
          <a
            href="#"
            className="flex items-center gap-3 text-white px-4 py-2 rounded-md hover:bg-hover-bg"
          >
            <Icon name="user" />
            <span>Users</span>
          </a>
        </div>
      </nav>
    </aside>
  );
};
