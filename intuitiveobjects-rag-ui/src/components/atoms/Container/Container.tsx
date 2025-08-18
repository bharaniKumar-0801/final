interface ContainerProps {
  children: React.ReactNode;
  className?: string;
}

export const Container = ({ children, className = "" }: ContainerProps) => {
  return <div className={`max-w-3xl mx-auto ${className}`}>{children}</div>;
};
