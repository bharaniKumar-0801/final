import { SelectHTMLAttributes } from "react";

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  className?: string;
}

export const Select = ({ className = "", ...props }: SelectProps) => {
  return (
    <select
      {...props}
      className={`bg-chat-bg border border-chat-border rounded-md px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-chat-border ${className}`}
    />
  );
};
