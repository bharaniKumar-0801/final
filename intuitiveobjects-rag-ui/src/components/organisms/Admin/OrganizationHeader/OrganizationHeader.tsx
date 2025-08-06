import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { TextInput } from "@/components/atoms/TextInput";

interface OrganizationSettings {
  domain: string;
}

export const OrganizationHeader = () => {
  const [settings, setSettings] = useState<OrganizationSettings>({
    domain: "",
  });

  const handleSaveDomain = () => {
    if (settings.domain.trim()) {
      console.log("Saving domain:", settings.domain);
      // Add your save logic here
    }
  };

  return (
    <div className="bg-sidebar-bg border-b border-chat-border p-6 mb-6">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-lg font-medium text-white mb-4">
          Organization Domain
        </h2>
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <TextInput
              label="Organization Domain"
              type="text"
              value={settings.domain}
              onChange={(e) =>
                setSettings({ ...settings, domain: e.target.value })
              }
              placeholder="Enter organization domain (e.g., company.com)"
            />
          </div>
          <div>
            <Button
              onClick={handleSaveDomain}
              disabled={!settings.domain.trim()}
            >
              Save Domain
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
