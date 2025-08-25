interface ChipProps {
  label: string;
  color: string;
  removeTag: (tag: string) => void;
}

const Chip = (props: ChipProps) => {
  return (
    <div className="flex items-center bg-blue-100 text-blue-800 px-2 py-1 rounded-md">
      <span className="text-sm">{props.label}</span>
      <button
        type="button"
        onClick={() => props.removeTag(props.label)}
        className="ml-1 focus:outline-none"
      >
        <span className="text-sm">X</span>
      </button>
    </div>
  );
};

export default Chip;
