import React from "react";
interface ButtonProps {
  label: string;
}

const Button: React.FC<ButtonProps> = ({ label }) => {
  return (
    <button
      type="submit"
      className="w-full px-4 py-2 text-white bg-[#292458] rounded-lg hover:bg-[#8381a3]"
    >
      {label}
    </button>
  );
};

export default Button;
