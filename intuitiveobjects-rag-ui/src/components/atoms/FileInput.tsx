import React, { useRef, forwardRef } from "react";

interface FileInputProps {
	label?: string;
	onChange: (file: File | null) => void;
	accept?: string;
	error?: string;
	placeholder?: string;
	className?: string;
	filename?: string;
}

export const FileInput = forwardRef<HTMLInputElement | null, FileInputProps>(
	({ label, onChange, accept = "*/*", error, placeholder = "Choose a file", className = "", filename }, ref) => {
		const fileInputRef = useRef<HTMLInputElement | null>(null);

		const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
			const file = e.target.files?.[0] || null;
			onChange(file);
		};

		const handleBrowseClick = () => {
			fileInputRef.current?.click();
		};

		const handleClearClick = (e: React.MouseEvent) => {
			e.stopPropagation();
			if (fileInputRef.current) {
				fileInputRef.current.value = "";
			}
			onChange(null);
		};

		return (
			<div className="flex flex-col">
				{label && <label className="mb-2 block text-sm font-medium text-gray-400">{label}</label>}

				<div
					className={`border bg-primary-100 ${
						error ? "border-red" : "border-gray-600"
					} rounded-md flex items-center overflow-hidden ${className}`}>
					<div onClick={handleBrowseClick} className="flex-1 flex items-center cursor-pointer px-3 py-2 min-h-[40px]">
						{filename ? (
							<div className="flex items-center justify-between w-full">
								<div className="flex items-center">
									<svg className="w-5 h-5 mr-2 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
										<path
											strokeLinecap="round"
											strokeLinejoin="round"
											strokeWidth={2}
											d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
										/>
									</svg>
									<span title={filename} className="truncate max-w-[180px] text-white">
										{filename}
									</span>
								</div>

								<button type="button" onClick={handleClearClick} className="text-gray-400 hover:text-gray-600 ml-2">
									<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
										<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
									</svg>
								</button>
							</div>
						) : (
							<div className="text-gray-500 flex items-center">
								<svg className="w-5 h-5 mr-2 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
									<path
										strokeLinecap="round"
										strokeLinejoin="round"
										strokeWidth={2}
										d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
									/>
								</svg>
								<span>{placeholder}</span>
							</div>
						)}
					</div>

					<div className="bg-primary-500 hover:bg-primary-600 text-white py-2 px-4 cursor-pointer" onClick={handleBrowseClick}>
						Browse
					</div>

					<input
						ref={(element) => {
							fileInputRef.current = element;
							if (ref && "current" in ref) {
								ref.current = element;
							}
						}}
						type="file"
						className="hidden"
						onChange={handleFileChange}
						accept={accept}
						required
					/>
				</div>

				{error && <p className="mt-1 text-xl text-red-600">{error}</p>}
			</div>
		);
	}
);
