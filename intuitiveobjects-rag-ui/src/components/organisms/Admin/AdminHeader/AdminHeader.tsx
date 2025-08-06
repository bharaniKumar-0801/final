import { useAppDispatch, useAppSelector } from "@/store/hooks";
import { adminLogout } from "@/store/slices/authSlice";
import { Menu, MenuButton, MenuItem, MenuItems } from "@headlessui/react";
import { useNavigate } from "react-router-dom";
import { confirmAction } from "@/utils/sweetAlert";

export const AdminHeader = () => {
	const dispatch = useAppDispatch();
	const navigate = useNavigate();

	const admin = useAppSelector((state) => state.auth.admin?.user);

	// Get initials for avatar
	const getInitials = () => {
		if (!admin?.name && !admin?.email) return "A";
		if (admin?.name) return admin.name.charAt(0).toUpperCase();
		return admin.email.charAt(0).toUpperCase();
	};

	const handleLogout = () => {
		confirmAction("Logout", "Are you sure you want to logout?", "Yes, logout").then((result) => {
			if (result.isConfirmed) {
				dispatch(adminLogout());
				navigate("/login");
			}
		});
	};

	return (
		<header className="bg-sidebar-bg border-b border-chat-border h-16 sticky top-0 z-40">
			<div className="mx-auto px-4 sm:px-6 h-full flex items-center justify-between">
				<div className="flex items-center space-x-4">
					{/* Responsive title */}
					<h1 className="text-white text-lg sm:text-xl font-semibold truncate">Admin Dashboard</h1>
				</div>

				<div className="flex items-center gap-2 sm:gap-4">
					<Menu as="div" className="relative">
						<MenuButton className="flex items-center gap-2 focus:outline-none">
							{/* Avatar - slightly larger on mobile for better touch target */}
							<div className="w-9 h-9 sm:w-8 sm:h-8 rounded-full bg-gray-600 flex items-center justify-center text-white font-medium">
								{admin?.avatar ? (
									<img src={admin.avatar} alt="User avatar" className="w-full h-full object-cover rounded-full" />
								) : (
									<span>{getInitials()}</span>
								)}
							</div>
							{/* Hide name on very small screens */}
							<span className="hidden sm:inline text-white">{admin?.name || admin?.email}</span>
						</MenuButton>
						<MenuItems className="absolute right-0 mt-2 w-48 origin-top-right rounded-md bg-sidebar-bg shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none z-10">
							{/* Show user info on mobile since name is hidden */}
							<div className="px-4 py-2 border-b border-gray-700 sm:hidden">
								<p className="text-sm text-gray-400">Signed in as</p>
								<p className="text-sm font-medium text-white truncate">{admin?.name || admin?.email}</p>
							</div>
							<MenuItem>
								<button
									className="text-white w-full text-left px-4 py-3 sm:py-2 text-sm hover:bg-chat-bg rounded-md flex items-center"
									onClick={handleLogout}>
									{/* Logout Icon */}
									<svg className="w-5 h-5 sm:w-4 sm:h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
										<path
											strokeLinecap="round"
											strokeLinejoin="round"
											strokeWidth="2"
											d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
										/>
									</svg>
									Logout
								</button>
							</MenuItem>
						</MenuItems>
					</Menu>
				</div>
			</div>
		</header>
	);
};
