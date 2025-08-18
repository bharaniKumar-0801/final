import { useEffect } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { TextInput } from "@/components/atoms/TextInput";
import useFormValidation from "@/hooks/useFormValidation";
import { validateCategoryForm } from "@/hooks/vaidate";
import { orgCreateCategoryApi, orgDeleteCategoryApi, orgGetCategoriesApi } from "@/services/adminApi";
import { useAppDispatch, useAppSelector } from "@/store/hooks";
import { TrashIcon } from "@/components/atoms/Icon/Trash";
import { confirmAction, showSuccess, showError } from "@/utils/sweetAlert";
import { ResponsiveTable } from "@/components/atoms/ResponsiveTable/ResponsiveTable";
import { toast } from "react-hot-toast";
const initialCategoryFormState = {
	name: "",
};

export const CategoryTab = () => {
	const dispatch = useAppDispatch();
	const categories = useAppSelector((state) => state.admin.categories);
	const isLoading = useAppSelector((state) => state.admin.loading);

	const categoryForm = useFormValidation(initialCategoryFormState, validateCategoryForm);

	const handleAddCategory = () => {
		dispatch(orgCreateCategoryApi(categoryForm.values));
		toast.success("Category added!");
		categoryForm.onReset();
	};

	const handleRemoveCategory = (id: string) => {
		confirmAction("Are you sure?", "You won't be able to revert this!", "Yes, delete it!").then((result) => {
			if (result.isConfirmed) {
				dispatch(orgDeleteCategoryApi(id)).then((response) => {
					if (response.meta.requestStatus === "fulfilled") {
						showSuccess("Deleted!", "Category has been deleted.");
					} else {
						showError("Error!", "Failed to delete category.");
					}
				});
			}
		});
	};

	useEffect(() => {
		dispatch(orgGetCategoriesApi());
	}, []);

	// Define columns for the responsive table
	const columns = [
		{
			key: "name",
			header: "Category Name",
			render: (value: string) => (
				<div className="flex items-center">
					<div className="text-sm font-medium text-gray-400 truncate max-w-xs">{value}</div>
				</div>
			),
		},
		{
			key: "id",
			header: "Actions",
			className: "text-right",
			render: (id: string) => (
				<button onClick={() => handleRemoveCategory(id)} className="text-red-600 hover:text-red-900 focus:outline-none">
					<TrashIcon />
				</button>
			),
		},
	];

	return (
		<>
			<h1 className="text-2xl font-semibold text-white mb-6">Category Management</h1>
			<div className="mb-8">
				<div className="flex flex-col sm:flex-row gap-4 sm:items-end">
					<div className="flex-1">
						<TextInput
							label="Category Name"
							type="text"
							name="name"
							error={categoryForm.errors.name}
							value={categoryForm.values.name}
							onChange={(e) => categoryForm.handleChange(e)}
							placeholder="New category name"
						/>
					</div>
					<Button
						onClick={() => categoryForm.handleSubmit(handleAddCategory)}
						disabled={!categoryForm.values.name.trim()}
						className="w-full sm:w-auto">
						Add Category
					</Button>
				</div>
			</div>

			<div className="space-y-4">
				<ResponsiveTable columns={columns} data={categories} emptyMessage="No categories added yet" isLoading={isLoading} />
			</div>
		</>
	);
};
