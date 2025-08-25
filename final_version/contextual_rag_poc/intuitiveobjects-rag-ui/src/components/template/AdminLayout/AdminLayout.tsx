import { ReactNode } from "react";
import { AdminHeader } from "@/components/organisms/Admin/AdminHeader/AdminHeader";
import { Outlet } from "react-router-dom";

interface AdminLayoutProps {
	children?: ReactNode;
}

export const AdminLayout = ({ children }: AdminLayoutProps) => {
	return (
		<div className="xl:min-h-screen bg-chat-bg">
			<AdminHeader />
			<div className="flex ">
				<div className="flex-1  p-8">{children || <Outlet />}</div>
			</div>
		</div>
	);
};
