import { TextareaHTMLAttributes, forwardRef } from "react";

interface TextAreaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  className?: string;
}

export const TextArea = forwardRef<HTMLTextAreaElement, TextAreaProps>(
  ({ className = "", ...props }, ref) => {
    return (
      <textarea
        ref={ref}
        {...props}
        className={`bg-chat-bg border border-chat-border rounded-md px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-chat-border resize-none ${className}`}
      />
    );
  }
);

TextArea.displayName = "TextArea";
