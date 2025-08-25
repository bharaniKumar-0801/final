import React, { ChangeEvent } from "react";

interface TextInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  type?: string;
  error?: string;
  label?: string;
  value?: string;
  placeholder?: string;
  name?: string;
  onChange: (e: ChangeEvent<HTMLInputElement>) => void;
  hideLabel?: boolean;
}

export const TextInput: React.FC<TextInputProps> = ({
  type = "text",
  error,
  label,
  value,
  placeholder,
  onChange,
  hideLabel = false,
  className,
  ...props
}) => {
  return (
    <div>
      {!hideLabel && (
        <label className="block text-sm font-medium text-gray-400 mb-2">
          {label}
        </label>
      )}
      <input
        type={type}
        value={value}
        placeholder={placeholder}
        onChange={onChange}
        className={`w-full bg-chat-bg border border-chat-border rounded-md px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-chat-border ${className}`}
        {...props}
      />
      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
};
