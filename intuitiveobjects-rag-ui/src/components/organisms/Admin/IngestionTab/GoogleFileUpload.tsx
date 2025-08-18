import useFormValidation from "@/hooks/useFormValidation";
import { validateIngestionForm } from "@/hooks/vaidate";
import { orgIngestDocumentApi } from "@/services/adminApi";
import { useAppSelector, useAppDispatch } from "@/store/hooks";
import { useEffect, useState } from "react";
import useDrivePicker from "react-google-drive-picker";
import toast from "react-hot-toast";
import UploadingFileProgress from "./UploadFilePreogress";
import { SelectInput } from "@/components/atoms/SelectInput";
import { Button } from "@/components/atoms/Button/Button";
import { TextInput } from "@/components/atoms/TextInput";
import Chip from "@/components/atoms/Chip";
import { formatFileSize } from "@/utils/function";
import { Loader } from "@/components/atoms/Loading/Loading";
import { GoogleDriveIcon } from "@/components/atoms/Icon/GoogleDriveIcon";
declare const gapi: any;

const initialState = {
	category_id: "",
	file: null,
	tags: [],
};

const toastId = "upload-toast";

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;
const GOOGLE_API_KEY = import.meta.env.VITE_GOOGLE_API_KEY;

const GoogleFileUpload = ({ setIsUploading, isUploading }: { setIsUploading: (isUploading: boolean) => void; isUploading: boolean }) => {
	const dispatch = useAppDispatch();
	const [openPicker] = useDrivePicker();
	const [isDriveFileLoading, setIsDriveFileLoading] = useState(false);

	const categories = useAppSelector((state) => state.admin.categories);
	const ingestionForm = useFormValidation(initialState, validateIngestionForm);
	const [error, setError] = useState<string | null>(null);

	const [newTag, setNewTag] = useState("");
	const isFormValid = () => {
		return ingestionForm.values.category_id.trim() !== "" && ingestionForm.values.file !== null && ingestionForm.values.tags.length > 0;
	};
	useEffect(() => {
		if (error) setError(null);
	}, [ingestionForm.values]);
	const onSubmit = async () => {
		if (!isFormValid()) {
			toast.error("Please select a category and upload a file");
			setError("Please select a category and upload a file");
			return;
		}
		setIsUploading(true);
		const response = await dispatch(
			orgIngestDocumentApi({
				values: ingestionForm.values,
				onProgress: (progress: number) => {
					toast.custom(
						() => UploadingFileProgress(ingestionForm.values.file?.name, formatFileSize(ingestionForm.values.file?.size), progress),
						{
							id: toastId,
							position: "bottom-right",
							duration: Infinity,
						}
					);
				},
			})
		);

		if (response.payload) {
			toast.success("Document uploaded successfully", {
				id: toastId,
				duration: 2000,
			});
			setIsUploading(false);
			ingestionForm.onReset();
			setNewTag("");
		}
	};

	const handleAddNewTag = () => {
		const isAlreadyInTags = ingestionForm.values.tags.find((tag: string) => tag === newTag);

		if (isAlreadyInTags) {
			toast.error("Tag already exists");
			return;
		}

		ingestionForm.setValues({
			...ingestionForm.values,
			tags: [...ingestionForm.values.tags, newTag],
		});
		setNewTag("");
	};

	const handleRemoveTag = (tag: string) => {
		ingestionForm.setValues({
			...ingestionForm.values,
			tags: ingestionForm.values.tags.filter((t: string) => t !== tag),
		});
	};

	const handleSelectCategory = (category: string) => {
		ingestionForm.setValues({
			...ingestionForm.values,
			category_id: category,
		});
	};

	const handleOpenPicker = () => {
		gapi.load("client:auth2", () => {
			gapi.client
				.init({
					apiKey: GOOGLE_API_KEY,
				})
				.then(() => {
					let tokenInfo = gapi.auth.getToken();
					const pickerConfig: any = {
						clientId: GOOGLE_CLIENT_ID,
						developerKey: GOOGLE_API_KEY,
						viewId: "DOCS",
						viewMimeTypes: "application/pdf",
						token: tokenInfo ? tokenInfo.access_token : null,
						showUploadView: false,
						showUploadFolders: false,
						supportDrives: false,
						multiselect: false,
						callbackFunction: async (data: any) => {
							setIsDriveFileLoading(true);
							if (data.action === "picked") {
								if (!tokenInfo) {
									tokenInfo = gapi.auth.getToken();
								}
								const fetchOptions = {
									headers: {
										Authorization: `Bearer ${tokenInfo.access_token}`,
									},
								};
								const driveFileUrl = "https://www.googleapis.com/drive/v3/files";
								const file = data.docs[0];
								const response = await fetch(`${driveFileUrl}/${file.id}?alt=media`, fetchOptions);
								const blob = await response.blob();
								const responseFile = new File([blob], file.name, {
									type: blob.type,
								});
								ingestionForm.setValues({
									...ingestionForm.values,
									file: responseFile,
								});
								setIsDriveFileLoading(false);
							}
						},
					};

					openPicker(pickerConfig);
				});
		});
	};

	return (
		<>
			<div className="bg-primary-100 rounded-lg">
				<h2 className="text-lg text-gray-200 font-medium mb-4">Upload Google Document</h2>
				<form
					onSubmit={(e) => {
						e.preventDefault();
						ingestionForm.handleSubmit(onSubmit);
					}}
					className="space-y-4">
					<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
						<SelectInput
							label="Category"
							name="category_id"
							value={ingestionForm.values.category_id}
							onChange={(e) => handleSelectCategory(e.target.value)}
							options={categories.map((category) => ({
								value: category.id,
								label: category.name,
							}))}
						/>

						<div className="xl:flex flex-col items-end sm:flex-row gap-2">
							<div className="flex-1">
								<div className="h-[42px] flex items-center gap-2 border border-gray-600 rounded-md p-2">
									{ingestionForm.values.file?.name && <p className="text-sm text-gray-200">{ingestionForm.values.file?.name}</p>}
								</div>
							</div>
							<Button type="button" onClick={handleOpenPicker} className="bg-primary-500 text-white p-2 h-fit mt-6 w-full sm:w-auto">
								<GoogleDriveIcon />
							</Button>
						</div>

						<div className="xl:flex flex-col sm:flex-row space-y-2 sm:space-y-0 gap-2 items-end">
							<div className="flex-1">
								<TextInput label="Tags" name="tags" value={newTag} onChange={(e) => setNewTag(e.target.value)} />
							</div>
							<Button
								type="button"
								onClick={handleAddNewTag}
								disabled={!newTag.trim()}
								className="bg-primary-500 text-white p-2 h-fit w-full sm:w-auto">
								Add Tag
							</Button>
						</div>
					</div>

					<div className="flex flex-wrap gap-2 mb-2">
						{ingestionForm.values?.tags?.map((tag: string) => (
							<Chip key={tag} label={tag} removeTag={handleRemoveTag} color="blue" />
						))}
					</div>

					<Button type="submit" disabled={isUploading || !isFormValid()} className="w-full sm:w-auto">
						{isUploading ? "Uploading..." : "Upload Document"}
					</Button>
				</form>
			</div>
			{isDriveFileLoading && (
				<div className="z-50 fixed bg-black/50 top-0 left-0 w-full h-screen flex items-center justify-center">
					<Loader />
				</div>
			)}
		</>
	);
};

export default GoogleFileUpload;
