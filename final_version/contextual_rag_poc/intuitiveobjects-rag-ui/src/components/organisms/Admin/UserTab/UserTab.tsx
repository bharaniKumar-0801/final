import { useState, useEffect } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { TextInput } from "@/components/atoms/TextInput";
import { SelectInput } from "@/components/atoms/SelectInput";
import { TrashIcon } from "@/components/atoms/Icon/Trash";
import toast from "react-hot-toast";
import { useAppDispatch, useAppSelector } from "@/store/hooks";
import useFormValidation from "@/hooks/useFormValidation";
import { validateUserForm } from "@/hooks/vaidate";
import { orgCreateUserApi, orgGetUsersApi, orgGetCategoriesApi, orgDeleteUserApi } from "@/services/adminApi";
import { showError, showSuccess, confirmAction } from "@/utils/sweetAlert";
import { ResponsiveTable } from "@/components/atoms/ResponsiveTable/ResponsiveTable";

const initialUserFormState = {
	email: "",
	password: "",
	category_id: "",
};

export const UserTab = () => {
	const dispatch = useAppDispatch();

	const categories = useAppSelector((state) => state.admin.categories);
	const users = useAppSelector((state) => state.admin.users);

	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);
	const userForm = useFormValidation(initialUserFormState, validateUserForm);

	// Clear error when form values change
	useEffect(() => {
		if (error) setError(null);
	}, [userForm.values]);

	const onSubmit = async () => {
		// Check if there are any validation errors by manually validating all fields
		const hasErrors =
			Object.keys(userForm.errors).length > 0 || !userForm.values.email || !userForm.values.password || !userForm.values.category_id;

		if (hasErrors) {
			// Force touch all fields to show errors
			const touchedFields: Record<string, boolean> = {};
			Object.keys(userForm.values).forEach((key) => {
				touchedFields[key] = true;
			});

			userForm.setValues({ ...userForm.values }); // Trigger validation
			return;
		}

		setLoading(true);
		try {
			const response = await dispatch(orgCreateUserApi(userForm.values));

			if (response.type === "org/createUser/fulfilled") {
				toast.success("User created successfully");
				userForm.onReset(); // Clear form after success
				dispatch(orgGetUsersApi()); // Refresh user list
			} else {
				// Handle API error
				const errorMessage = response.payload?.response?.data?.message || response.payload?.error || "Failed to create user";

				toast.error(errorMessage);
				setError(errorMessage);
			}
		} catch (err: any) {
			const errorMessage = err?.message || "An unexpected error occurred";
			toast.error(errorMessage);
			setError(errorMessage);
		} finally {
			setLoading(false);
		}
	};

	const handleRemoveUser = async (id: string) => {
		confirmAction("Are you sure?", "This user will be permanently deleted. You won't be able to revert this!", "Yes, delete user!").then(
			(result: any) => {
				if (result.isConfirmed) {
					dispatch(orgDeleteUserApi(id)).then((response) => {
						if (response.type === "org/deleteUser/fulfilled") {
							showSuccess("Deleted!", "User has been deleted successfully.");
						} else {
							showError("Error!", "Failed to delete user.");
						}
					});
				}
			}
		);
	};

	// Load data when component mounts
	useEffect(() => {
		dispatch(orgGetUsersApi());
		dispatch(orgGetCategoriesApi());
	}, []);

	// Define columns for responsive table
	const columns = [
		{
			key: "email",
			header: "Email",
			render: (value: string) => (
				<div className="flex items-center">
					<div className="text-sm font-medium text-gray-400 truncate max-w-[150px] sm:max-w-xs">{value}</div>
				</div>
			),
		},
		{
			key: "is_active",
			header: "Status",
			render: (value: boolean) => (
				<div className="flex items-center">
					<div
						className={`px-2 py-0.5 inline-flex text-xs leading-5 font-semibold rounded-md ${
							value ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
						}`}>
						{value ? "Active" : "Inactive"}
					</div>
				</div>
			),
		},
		{
			key: "id",
			header: "Actions",
			className: "text-right",
			render: (id: string) => (
				<button onClick={() => handleRemoveUser(id)} className="text-red-600 hover:text-red-900 focus:outline-none">
					<TrashIcon />
				</button>
			),
		},
	];

	const isFormValid = () => {
		return userForm.values.email.trim() !== "" && userForm.values.password.trim() !== "" && userForm.values.category_id.trim() !== "";
	};

	return (
		<>
			<h1 className="text-2xl text-gray-400 font-semibold mb-6">User Management</h1>

			{error && <div className="bg-red-100 border border-red-400 text-red-700 p-3 rounded-md mb-4">{error}</div>}

			<div className="bg-primary-100 xl:p-4 sm:p-6 rounded-lg mb-8">
				<h2 className="text-lg text-gray-200 font-medium mb-4">Add New User</h2>
				<form
					onSubmit={(e) => {
						e.preventDefault();
						userForm.handleSubmit(onSubmit);
					}}
					className="space-y-4">
					<div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
						<TextInput
							label="Email"
							type="email"
							name="email"
							value={userForm.values.email}
							onChange={userForm.handleChange}
							error={userForm.errors.email}
							placeholder="Enter email"
							required
						/>
						<TextInput
							label="Password"
							type="password"
							name="password"
							value={userForm.values.password}
							onChange={userForm.handleChange}
							error={userForm.errors.password}
							placeholder="Enter password"
							required
						/>
						<SelectInput
							label="Category"
							name="category_id"
							value={userForm.values.category_id}
							onChange={(e) => {
								userForm.handleChange(e);
							}}
							error={userForm.errors.category_id}
							options={categories.map((category) => ({
								value: category.id,
								label: category.name,
							}))}
							required
						/>
					</div>
					<Button type="submit" disabled={loading || !isFormValid()} className="w-full sm:w-auto mt-2">
						{loading ? "Adding..." : "Add User"}
					</Button>
				</form>
			</div>

			<div className="mb-8">
				<h2 className="text-lg text-gray-200 font-medium mb-4">User Accounts</h2>

				<ResponsiveTable columns={columns} data={users} emptyMessage="No users added yet" isLoading={loading && users.length === 0} />
			</div>
		</>
	);
};
