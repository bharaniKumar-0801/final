import { useState } from "react";
import { TabContent } from "@/components/molecules/TabContent/TabContent";
import { CategoryTab } from "@/components/organisms/Admin/CategoryTab/CategoryTab";
import { UserTab } from "@/components/organisms/Admin/UserTab/UserTab";
import { IngestionTab } from "@/components/organisms/Admin/IngestionTab/IngestionTab";
import { SettingsTab } from "@/components/organisms/Admin/SettingsTab/SettingsTab";
import { Menu, MenuButton, MenuItem, MenuItems } from "@headlessui/react";
import AdditionalTab from "@/components/organisms/Admin/AdditionalTab/additionalTab";

export const Admin = () => {
	const [activeTab, setActiveTab] = useState("categories");

	const tabs = [
		{ id: "categories", label: "Categories" },
		{ id: "users", label: "Users" },
		{ id: "ingestion", label: "Ingestion" },
		{ id: "settings", label: "Settings" },
		{ id: "additional", label: "Additional" },
	];

	const handleTabChange = (tabId: string) => {
		setActiveTab(tabId);
	};

	return (
		<div className="w-full max-w-4xl mx-auto">
			{/* Desktop Tab Navigation */}
			<div className="hidden sm:block mb-6 border-b border-chat-border">
				<div className="flex flex-wrap">
					{tabs.map((tab) => (
						<button
							key={tab.id}
							onClick={() => handleTabChange(tab.id)}
							className={`px-4 py-2 text-sm font-medium focus:outline-none ${
								activeTab === tab.id
									? "text-white border-b-2 border-white"
									: "text-gray-400 hover:text-white hover:border-b-2 hover:border-gray-400"
							}`}>
							{tab.label}
						</button>
					))}
				</div>
			</div>

			{/* Mobile Dropdown for Tabs */}
			<div className="sm:hidden mb-6">
				<Menu as="div" className="relative">
					<MenuButton className="w-full flex items-center justify-between px-4 py-2 bg-sidebar-bg rounded-md border border-gray-700 text-white">
						<span>{tabs.find((tab) => tab.id === activeTab)?.label || "Select Tab"}</span>
						<svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
						</svg>
					</MenuButton>
					<MenuItems className="absolute left-0 right-0 mt-1 origin-top-right rounded-md bg-sidebar-bg shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none z-10">
						{tabs.map((tab) => (
							<MenuItem key={tab.id}>
								{({ active }) => (
									<button
										className={`${
											active || activeTab === tab.id ? "bg-gray-700" : ""
										} w-full text-left px-4 py-3 text-sm text-white`}
										onClick={() => handleTabChange(tab.id)}>
										{tab.label}
									</button>
								)}
							</MenuItem>
						))}
					</MenuItems>
				</Menu>
			</div>

			<div className="bg-sidebar-bg rounded-lg p-4 sm:p-6 shadow-lg">
				<TabContent id="categories" activeTab={activeTab}>
					<CategoryTab />
				</TabContent>
				<TabContent id="users" activeTab={activeTab}>
					<UserTab />
				</TabContent>
				<TabContent id="ingestion" activeTab={activeTab}>
					<IngestionTab />
				</TabContent>
				<TabContent id="settings" activeTab={activeTab}>
					<SettingsTab />
				</TabContent>
				<TabContent id="additional" activeTab={activeTab}>
                    <AdditionalTab />
                </TabContent>
			</div>
		</div>
	);
};
