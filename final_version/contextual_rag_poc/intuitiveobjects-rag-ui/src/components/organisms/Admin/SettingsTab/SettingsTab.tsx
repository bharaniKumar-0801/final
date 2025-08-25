import { useEffect, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Select } from "@/components/atoms/Select/Select";
import { TextArea } from "@/components/atoms/TextArea/TextArea";
import { useAppDispatch, useAppSelector } from "@/store/hooks";
import {
  settingGetAppConfigApi,
  settingUpdateAppConfigApi,
  settingGetAppCurrentConfigApi,
} from "@/services/adminApi";
import { toast } from "react-hot-toast";

interface SystemSettings {
  llm_model: string;
  embedding_model: string;
  system_prompt: string;
  temperature: number;
}

export const SettingsTab = () => {
  const [settings, setSettings] = useState<SystemSettings>({
    llm_model: "",
    embedding_model: "",
    system_prompt: "",
    temperature: 0.0,
  });

  const handleSaveSettings = () => {
    dispatch(settingUpdateAppConfigApi(settings));
    toast.success("Settings saved successfully");
    setSettings({
      llm_model: "",
      embedding_model: "",
      system_prompt: "",
      temperature: 0.0,
    });
  };

  const dispatch = useAppDispatch();
  const { appConfig, currentAppConfig } = useAppSelector(
    (state: any) => state.admin
  );
  console.log(appConfig);

  useEffect(() => {
    dispatch(settingGetAppConfigApi());
    dispatch(settingGetAppCurrentConfigApi());
  }, []);

  const isFormValid = () => {
    return (
      settings.llm_model.trim() !== "" &&
      settings.embedding_model.trim() !== "" &&
      settings.system_prompt.trim() !== "" &&
      settings.temperature >= 0.0 &&
      settings.temperature <= 1.0
    );
  };

  return (
    <>
      <h1 className="text-2xl font-semibold text-white mb-4 sm:mb-6">
        System Settings
      </h1>

      <div className="space-y-6 sm:space-y-8">
        {/* LLM Model Selection */}
        <div className="bg-primary-100 rounded-lg">
          <h2 className="text-lg font-medium text-white mb-2">LLM Model</h2>
          <Select
            value={settings.llm_model}
            onChange={(e) =>
              setSettings({ ...settings, llm_model: e.target.value })
            }
            className="w-full"
          >
            <option value="">Select LLM model</option>
            <option value="qwen">qwen2.5:1.5b</option>
            <option value="phi">phi3.5:latest</option>
            <option value="llama">llama3.1:70b</option>
            <option value="gemma">gemma2:2b</option>
            <option value="deepseekcoder">deepseek-coder:1.3b</option>
            {appConfig &&
              appConfig.length &&
              appConfig[0]?.llm_models.map((option: any) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
          </Select>
        </div>

        {/* Embedding Model Selection */}
        <div className="bg-primary-100 rounded-lg">
          <h2 className="text-lg font-medium text-white mb-2">
            Embedding Model (Optional)
          </h2>
          <Select
            value={settings.embedding_model}
            onChange={(e) =>
              setSettings({ ...settings, embedding_model: e.target.value })
            }
            className="w-full"
          >
            <option value="">Select embedding model</option>
            <option value="all-MiniLM-L6-v2">all-MiniLM-L6-v2</option>
            {appConfig &&
              appConfig.length &&
              appConfig[0]?.embedding_models?.map((option: any) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
          </Select>
        </div>

        {/* System Prompt */}
        <div className="bg-primary-100 rounded-lg">
          <h2 className="text-lg font-medium text-white mb-2">System Prompt</h2>
          <TextArea
            value={settings.system_prompt}
            onChange={(e) =>
              setSettings({ ...settings, system_prompt: e.target.value })
            }
            placeholder="Enter system prompt..."
            rows={6}
            className="w-full"
          />
        </div>
        
        {/* Temperature Setting */}   
        <div className="bg-primary-100 rounded-lg">
          <h2 className="text-lg font-medium text-white mb-2">Temperature</h2>
          <input
            type="number"
            value={settings.temperature}
            onChange={(e) =>
              setSettings({
                ...settings,
                temperature: parseFloat(e.target.value),
              })
            }
            min="0"
            max="1"
            step="0.1"
            placeholder="Enter temperature (0.0 - 1.0)"
            className="w-full p-2 bg-chat-bg text-white rounded-lg border border-chat-border focus:outline-none focus:ring-2 focus:ring-primary-500"
          />
        </div>

        {/* Save Button */}
        <div>
          <Button
            onClick={handleSaveSettings}
            disabled={!isFormValid()}
            className="w-full sm:w-auto"
          >
            Save Settings
          </Button>
        </div>
      </div>

      {/* Current Settings Display */}
      <div className="mt-6 sm:mt-8 p-4 sm:p-5 bg-chat-bg rounded-lg border border-chat-border">
        <h3 className="text-white font-medium mb-2">Current Configuration</h3>
        <div className="space-y-2 text-gray-400">
          <div className="flex flex-wrap items-center">
            <span className="w-44 sm:w-32 font-medium">LLM Model:</span>
            <span>{currentAppConfig?.llm_model || "Not set"}</span>
          </div>
          <div className="flex flex-wrap items-center">
            <span className="w-44 sm:w-32 font-medium">Embedding Model:</span>
            <span>{currentAppConfig?.embedding_model || "Not set"}</span>
          </div>
          <div className="flex flex-wrap items-center">
            <span className="w-44 sm:w-32 font-medium">System Prompt:</span>
            <span>{currentAppConfig?.system_prompt || "Not set"}</span>
          </div>
          <div className="flex flex-wrap items-center">
            <span className="w-44 sm:w-32 font-medium">Temperature:</span>
            <span>{currentAppConfig?.temperature || "Not set"}</span>
          </div>  
        </div>
      </div>
    </>
  );
};
