import { Container } from "../Container/Container";

interface LoadingProps {
  message?: string;
}

export const Loading = ({
  message = "Assistant is typing...",
}: LoadingProps) => {
  return <div className="p-6 text-gray-400 text-center">{message}</div>;
};

export const Loader = ({ className }: { className?: string }) => {
  return (
    <Container>
      <div
        className={`flex justify-center py-[4rem] items-center ${className}`}
      >
        <div className="spinner"></div>
      </div>
    </Container>
  );
};
