interface TabContentProps {
  id: string;
  activeTab: string;
  children: React.ReactNode;
}

export const TabContent = ({ id, activeTab, children }: TabContentProps) => {
  if (activeTab !== id) return null;
  return <>{children}</>;
};
