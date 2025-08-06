interface PersonIconProps {
  size?: number;
  className?: string;
}

const PersonIcon = ({ size = 24, className }: PersonIconProps) => {
  return (
    <svg
      focusable="false"
      aria-hidden="true"
      viewBox="0 0 24 24"
      width={size}
      height={size}
      className={className}
    >
      <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4m0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4"></path>
    </svg>
  );
};

export default PersonIcon;
