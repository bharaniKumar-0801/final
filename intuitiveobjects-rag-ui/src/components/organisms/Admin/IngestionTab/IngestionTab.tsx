import { useEffect, useState } from "react";
import { TrashIcon } from "@/components/atoms/Icon/Trash";
import { useAppSelector, useAppDispatch } from "@/store/hooks";
import { orgDeleteFileApi, orgGetCategoriesApi, orgGetFilesApi } from "@/services/adminApi";
import PdfIcon from "@/components/atoms/Icon/PdfIcon";
import { confirmAction } from "@/utils/sweetAlert";
import { ResponsiveTable } from "@/components/atoms/ResponsiveTable/ResponsiveTable";
import toast from "react-hot-toast";
import GoogleFileUpload from "./GoogleFileUpload";
import LocalFileUpload from "./LocalFileUpload";

export const IngestionTab = () => {
	const dispatch = useAppDispatch();
	const [isUploading, setIsUploading] = useState(false);

	const categories = useAppSelector((state) => state.admin.categories);
	const files = useAppSelector((state) => state.admin.files);

	const formatFileSize = (sizeInBytes: number) => {
		if (sizeInBytes < 1024) return `${sizeInBytes} B`;
		if (sizeInBytes < 1024 * 1024) return `${(sizeInBytes / 1024).toFixed(1)} KB`;
		return `${(sizeInBytes / (1024 * 1024)).toFixed(1)} MB`;
	};

	// Format date to be human-readable
	const formatDate = (dateString: string) => {
		const date = new Date(dateString);
		return date.toLocaleDateString("en-US", {
			year: "numeric",
			month: "short",
			day: "numeric",
			hour: "2-digit",
			minute: "2-digit",
		});
	};

	// Add this function to get the category name from ID
	const getCategoryName = (categoryId: string) => {
		const category = categories.find((cat) => cat.id === categoryId);
		return category ? category.name : "Unknown Category";
	};

	const handleDeleteFile = async (fileId: string) => {
		confirmAction("Are you sure?", "This file will be permanently deleted. You won't be able to revert this!", "Yes, delete file!").then(
			async (result: any) => {
				if (result.isConfirmed) {
					const response = await dispatch(orgDeleteFileApi(fileId));
					if (response.payload) {
						toast.success("File deleted successfully");
					} else {
						toast.error("Failed to delete file");
					}
				}
			}
		);
	};

	useEffect(() => {
		dispatch(orgGetCategoriesApi());
	}, []);

	useEffect(() => {
		dispatch(orgGetFilesApi());
	}, []);

	// Define columns for the responsive table
	const columns = [
		{
			key: "file_name",
			header: "File Name",
			render: (value: string) => (
				<div className="flex items-center gap-x-3">
					<span className="size-8 flex justify-center items-center border border-gray-200 text-gray-500 rounded-lg">
						<PdfIcon />
					</span>
					<div className="text-sm font-medium text-gray-400 truncate max-w-[150px] sm:max-w-xs">{value}</div>
				</div>
			),
		},
		{
			key: "category_id",
			header: "Category",
			render: (value: string) => (
				<span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-gray-100 text-gray-800">
					{getCategoryName(value)}
				</span>
			),
		},
		{
			key: "file_size",
			header: "Size",
			render: (value: number) => formatFileSize(value),
		},
		{
			key: "tags",
			header: "Tags",
			render: (tags: string[]) => {
				if (!tags || tags.length === 0) {
					return <span className="text-xs text-gray-400">No tags</span>;
				}

				// Display first 3 tags and a "+X more" indicator if there are more
				const displayLimit = 1;
				const displayTags = tags.slice(0, displayLimit);
				const remainingCount = Math.max(0, tags.length - displayLimit);

				return (
					<div className="flex flex-wrap gap-1 max-w-[200px]">
						{displayTags.map((tag, index) => (
							<span key={index} className="px-2 py-0.5 text-xs rounded-full bg-blue-100 text-blue-800 whitespace-nowrap" title={tag}>
								{tag}
							</span>
						))}
						{remainingCount > 0 && (
							<span
								className="px-2 py-0.5 text-xs rounded-full bg-gray-100 text-gray-700 whitespace-nowrap cursor-help"
								title={tags.slice(displayLimit).join(", ")}>
								+{remainingCount} more
							</span>
						)}
					</div>
				);
			},
		},
		{
			key: "created_at",
			header: "Date",
			render: (value: string) => formatDate(value),
		},
		{
			key: "id",
			header: "Actions",
			className: "text-right",
			render: (id: string) => (
				<button onClick={() => handleDeleteFile(id)} className="text-red-600 hover:text-red-900 focus:outline-none">
					<TrashIcon />
				</button>
			),
		},
	];

	return (
		<>
			<h1 className="text-2xl text-gray-200 font-semibold mb-6">Document Ingestion</h1>

			<div className="">
				<LocalFileUpload setIsUploading={setIsUploading} isUploading={isUploading} />
				<div className="mt-4">
					<GoogleFileUpload setIsUploading={setIsUploading} isUploading={isUploading} />
				</div>
			</div>

			<div className="my-8">
				<h2 className="text-lg text-gray-200 font-medium mb-4">Uploaded Documents</h2>
				<ResponsiveTable columns={columns} data={files} emptyMessage="No documents uploaded yet" isLoading={isUploading} />
			</div>
		</>
	);
};
