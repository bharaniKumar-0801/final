import React from "react";

interface SelectInputProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
	label?: string;
	error?: string;
	options: { value: string; label: string }[];
}

export const SelectInput: React.FC<SelectInputProps> = ({ label, error, options, className = "", ...props }) => {
	return (
		<div className="flex flex-col">
			{label && <label className="mb-2 block text-sm font-medium text-gray-400">{label}</label>}
			<select
				className={`px-3 py-2.5 border bg-transparent ${
					error ? "border-red-600" : "border-gray-600"
				} rounded-md focus:outline-none focus:ring-2 focus:ring-gray-600  text-white ${className}`}
				{...props}>
				<option className="text-gray-400" value="">
					Select...
				</option>
				{options.map((option) => (
					<option className="text-gray-400" key={option.value} value={option.value}>
						{option.label}
					</option>
				))}
			</select>
			{error && <p className="mt-1 text-xl text-red-600">{error}</p>}
		</div>
	);
};
