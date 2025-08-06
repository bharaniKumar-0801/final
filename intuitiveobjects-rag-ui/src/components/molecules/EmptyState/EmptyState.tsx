interface EmptyStateProps {
  title: string;
  description: string;
}

export const EmptyState = ({ title, description }: EmptyStateProps) => {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center text-gray-400">
        <h2 className="text-2xl font-semibold mb-4">{title}</h2>
        <p>{description}</p>
      </div>
    </div>
  );
};
