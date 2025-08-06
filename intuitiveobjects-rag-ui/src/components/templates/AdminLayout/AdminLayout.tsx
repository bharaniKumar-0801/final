import { useState, useEffect } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { AdminHeader } from "@/components/organisms/Admin/AdminHeader/AdminHeader";
import { Toaster } from "react-hot-toast";

export const AdminLayout = () => {
	const location = useLocation();
	const [activeTab, setActiveTab] = useState("category");
	const [sidebarOpen, setSidebarOpen] = useState(false);

	// Close sidebar when changing routes on mobile
	useEffect(() => {
		setSidebarOpen(false);
	}, [location.pathname]);

	const handleTabChange = (tab: string) => {
		setActiveTab(tab);
	};

	return (
		<div className="min-h-screen bg-primary-50 flex flex-col">
			<AdminHeader />
			<Toaster position="top-right" />

			<div className="flex flex-1 relative">
				{/* Mobile sidebar overlay */}
				{sidebarOpen && <div className="fixed inset-0 bg-black bg-opacity-50 z-20 md:hidden" onClick={() => setSidebarOpen(false)}></div>}

				{/* Sidebar Navigation - mobile drawer and desktop sidebar */}
				<div
					className={`${
						sidebarOpen ? "translate-x-0" : "-translate-x-full"
					} fixed inset-y-0 left-0 transform md:relative md:translate-x-0 z-30
            w-64 bg-primary-100 border-r border-gray-700 transition duration-300 ease-in-out
            pt-16 md:pt-0 overflow-y-auto md:block`}>
					<nav className="p-4">
						<ul className="space-y-2">
							<li>
								<Link
									to="/admin"
									className={`block px-4 py-2 rounded-md ${
										activeTab === "category" ? "bg-primary-500 text-white" : "text-gray-300 hover:bg-primary-200"
									}`}
									onClick={() => handleTabChange("category")}>
									Categories
								</Link>
							</li>
							<li>
								<Link
									to="/admin?tab=users"
									className={`block px-4 py-2 rounded-md ${
										activeTab === "users" ? "bg-primary-500 text-white" : "text-gray-300 hover:bg-primary-200"
									}`}
									onClick={() => handleTabChange("users")}>
									Users
								</Link>
							</li>
							<li>
								<Link
									to="/admin?tab=ingestion"
									className={`block px-4 py-2 rounded-md ${
										activeTab === "ingestion" ? "bg-primary-500 text-white" : "text-gray-300 hover:bg-primary-200"
									}`}
									onClick={() => handleTabChange("ingestion")}>
									Document Ingestion
								</Link>
							</li>
						</ul>
					</nav>
				</div>

				{/* Main Content */}
				<div className="flex-1 p-4 md:p-8 overflow-auto">
					<Outlet />
				</div>
			</div>
		</div>
	);
};
