interface AvatarProps {
  role: "user" | "assistant";
  className?: string;
}

export const Avatar = ({ role, className = "" }: AvatarProps) => {
  return (
    <div
      className={`w-8 h-8 rounded-sm bg-chat-border flex items-center justify-center text-white ${className}`}
    >
      {role === "user" ? "👤" : "🤖"}
    </div>
  );
};
