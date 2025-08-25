interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost";
  children: React.ReactNode;
}

export const Button = ({
  variant = "primary",
  children,
  className = "",
  ...props
}: ButtonProps) => {
  const baseStyles =
    "px-4 py-[9px] rounded-md transition-colors duration-200 disabled:opacity-50";
  const variantStyles = {
    primary:
      "bg-chat-border text-white hover:bg-hover-bg disabled:hover:bg-chat-border",
    secondary:
      "border border-chat-border text-white hover:bg-hover-bg disabled:hover:bg-transparent",
    ghost: "text-white hover:bg-hover-bg",
  };

  return (
    <button
      className={`${baseStyles} ${variantStyles[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
};
