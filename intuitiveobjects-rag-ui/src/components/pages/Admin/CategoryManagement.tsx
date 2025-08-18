import { useState } from "react";
import { TabContent } from "@/components/molecules/TabContent/TabContent";
import { CategoryTab } from "@/components/organisms/Admin/CategoryTab/CategoryTab";
import { UserTab } from "@/components/organisms/Admin/UserTab/UserTab";
import { IngestionTab } from "@/components/organisms/Admin/IngestionTab/IngestionTab";
import { SettingsTab } from "@/components/organisms/Admin/SettingsTab/SettingsTab";
import AdditionalTab from "@/components/organisms/Admin/AdditionalTab/additionalTab";


export const CategoryManagement = () => {
  const [activeTab, setActiveTab] = useState("categories");

  const tabs = [
    { id: "categories", label: "Categories" },
    { id: "users", label: "Users" },
    { id: "ingestion", label: "Ingestion" },
    { id: "settings", label: "Settings" },
    { id: "additional", label: "Additional" }
  ];

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-6 border-b border-chat-border">
        <div className="flex space-x-4">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 text-sm font-medium ${
                activeTab === tab.id
                  ? "text-white border-b-2 border-white"
                  : "text-gray-400 hover:text-white hover:border-b-2 hover:border-gray-400"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-sidebar-bg rounded-lg p-6 shadow-lg">
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
