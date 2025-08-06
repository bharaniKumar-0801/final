// import { useEffect, useRef, useState } from "react";
// import toast from "react-hot-toast";
// import useFormValidation from "@/hooks/useFormValidation";
// import { validateIngestionForm } from "@/hooks/vaidate";
// import { orgIngestDocumentApi} from "@/services/adminApi";
// import { useAppSelector, useAppDispatch } from "@/store/hooks";
// import UploadingFileProgress from "./UploadFilePreogress";
// import { SelectInput } from "@/components/atoms/SelectInput";
// import { FileInput } from "@/components/atoms/FileInput";
// import { Button } from "@/components/atoms/Button/Button";
// import { TextInput } from "@/components/atoms/TextInput";
// import Chip from "@/components/atoms/Chip";
// import { formatFileSize } from "@/utils/function";

// const initialState = {
//   category_id: "",
//   file: null,
//   tags: [],
// };

// const toastId = "upload-toast";

// const LocalFileUpload = ({
//   setIsUploading,
//   isUploading,
// }: {
//   setIsUploading: (isUploading: boolean) => void;
//   isUploading: boolean;
// }) => {
//   const dispatch = useAppDispatch();
//   const fileInputRef = useRef<HTMLInputElement | null>(null);
//   const categories = useAppSelector((state) => state.admin.categories);
//   const ingestionForm = useFormValidation(initialState, validateIngestionForm);

//   const [error, setError] = useState<string | null>(null);
//   const [newTag, setNewTag] = useState("");

//   const isFormValid = () =>
//     ingestionForm.values.category_id.trim() !== "" &&
//     ingestionForm.values.file !== null &&
//     ingestionForm.values.tags.length > 0;

//   useEffect(() => {
//     if (error) setError(null);
//   }, [ingestionForm.values]);

//   const onSubmit = async () => {
//     if (!isFormValid()) {
//       toast.error("Please select a category and upload a file");
//       setError("Please select a category and upload a file");
//       return;
//     }

//     setIsUploading(true);
//     const response = await dispatch(
//       orgIngestDocumentApi({
//         values: ingestionForm.values,
//         onProgress: (progress: number) => {
//           toast.custom(
//             () =>
//               UploadingFileProgress(
//                 ingestionForm.values.file?.name,
//                 formatFileSize(ingestionForm.values.file?.size),
//                 progress
//               ),
//             {
//               id: toastId,
//               position: "bottom-right",
//               duration: Infinity,
//             }
//           );
//         },
//       })
//     );

//     if (response.payload) {
//       toast.success("Document uploaded successfully", {
//         id: toastId,
//         duration: 2000,
//       });
//       setIsUploading(false);
//       ingestionForm.onReset();
//       setNewTag("");
//       if (fileInputRef.current) fileInputRef.current.value = "";
//     }
//   };

//   const handleAddNewTag = () => {
//     if (ingestionForm.values.tags.includes(newTag)) {
//       toast.error("Tag already exists");
//       return;
//     }

//     ingestionForm.setValues({
//       ...ingestionForm.values,
//       tags: [...ingestionForm.values.tags, newTag],
//     });
//     setNewTag("");
//   };

//   const handleRemoveTag = (tag: string) => {
//     ingestionForm.setValues({
//       ...ingestionForm.values,
//       tags: (ingestionForm.values.tags ?? []).filter((t:string) => t !== tag),
//     });
//   };


//   return (
//     <div className="bg-primary-100 rounded-lg border-b border-gray-700 pb-6 relative">


//       {/* Main Upload Form */}
//       <h2 className="text-lg text-gray-200 font-medium mb-4">Upload Local Document</h2>
//       <form
//         onSubmit={(e) => {
//           e.preventDefault();
//           ingestionForm.handleSubmit(onSubmit);
//         }}
//         className="space-y-4"
//       >
//         <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//           <SelectInput
//             label="Category"
//             name="category_id"
//             value={ingestionForm.values.category_id}
//             onChange={(e) =>
//               ingestionForm.setValues({
//                 ...ingestionForm.values,
//                 category_id: e.target.value,
//               })
//             }
//             options={categories.map((category) => ({
//               value: category.id,
//               label: category.name,
//             }))}
//           />

//           <FileInput
//             ref={fileInputRef}
//             label="Document File"
//             onChange={(file) => ingestionForm.setValues({ ...ingestionForm.values, file })}
//             filename={ingestionForm.values.file?.name}
//             accept=".pdf,.doc,.docx,.txt,.csv,.xls,.xlsx"
//             placeholder="Select a document"
//           />

//           <div className="xl:flex flex-col sm:flex-row space-y-2 sm:space-y-0 gap-2 items-end">
//             <TextInput
//               label="Tags"
//               name="tags"
//               value={newTag}
//               onChange={(e) => setNewTag(e.target.value)}
//             />
//             <Button type="button" onClick={handleAddNewTag} disabled={!newTag.trim()}>
//               Add Tag
//             </Button>
//           </div>
//         </div>

//         <div className="flex flex-wrap gap-2 mb-2">
//           {ingestionForm.values.tags.map((tag: string) => (
//             <Chip key={tag} label={tag} removeTag={handleRemoveTag} color="blue" />
//           ))}
//         </div>

//         <div className="flex flex-col sm:flex-row gap-2">
//           <Button type="submit" disabled={isUploading || !isFormValid()} className="w-full sm:w-auto">
//             {isUploading ? "Uploading..." : "Upload Document"}
//           </Button>
          
//         </div>
//       </form>
//     </div>
//   );
// };

// export default LocalFileUpload;



import useFormValidation from "@/hooks/useFormValidation";
import { validateIngestionForm } from "@/hooks/vaidate";
import { orgIngestDocumentApi } from "@/services/adminApi";
import { useAppSelector, useAppDispatch } from "@/store/hooks";
import { useEffect, useRef, useState } from "react";
import toast from "react-hot-toast";
import UploadingFileProgress from "./UploadFilePreogress";
import { SelectInput } from "@/components/atoms/SelectInput";
import { FileInput } from "@/components/atoms/FileInput";
import { Button } from "@/components/atoms/Button/Button";
import { TextInput } from "@/components/atoms/TextInput";
import Chip from "@/components/atoms/Chip";
import { formatFileSize } from "@/utils/function";

const initialState = {
  category_id: "",
  file: null,
  tags: [],
};

const toastId = "upload-toast";

const LocalFileUpload = ({
  setIsUploading,
  isUploading,
}: {
  setIsUploading: (isUploading: boolean) => void;
  isUploading: boolean;
}) => {
  const dispatch = useAppDispatch();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const categories = useAppSelector((state) => state.admin.categories);
  const ingestionForm = useFormValidation(initialState, validateIngestionForm);

  const [error, setError] = useState<string | null>(null);
  const [newTag, setNewTag] = useState("");

  const isFormValid = () =>
    ingestionForm.values.category_id.trim() !== "" &&
    ingestionForm.values.file !== null &&
    ingestionForm.values.tags.length > 0;

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

    let isUploadComplete = false;

    const response = await dispatch(
      orgIngestDocumentApi({
        values: ingestionForm.values,
        onProgress: (progress, info) => {
          if (isUploadComplete) return;

          toast.custom(
            () => (
              <UploadingFileProgress
                filename={ingestionForm.values.file?.name}
                size={formatFileSize(ingestionForm.values.file?.size)}
                progress={progress}
                completed={info?.completed}
                total={info?.total}
              />
            ),
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
      isUploadComplete = true;
      toast.dismiss(toastId);

      toast.success("Document uploaded successfully", {
        id: toastId,
        duration: 3000,
      });

      setIsUploading(false);
      ingestionForm.onReset();
      setNewTag("");

      setTimeout(() => {
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }, 100);
    } else {
      setIsUploading(false);
      toast.error("Upload failed. Please try again.");
    }
  };

  const handleAddNewTag = () => {
    if (
      newTag.trim() === "" ||
      ingestionForm.values.tags.includes(newTag.trim())
    ) {
      toast.error("Tag already exists or is invalid");
      return;
    }

    ingestionForm.setValues({
      ...ingestionForm.values,
      tags: [...ingestionForm.values.tags, newTag.trim()],
    });
    setNewTag("");
  };

  const handleRemoveTag = (tag: string) => {
    ingestionForm.setValues({
      ...ingestionForm.values,
      tags: ingestionForm.values.tags.filter((t: string) => t !== tag),
    });
  };

  const handleSelectFile = (file: File | null) => {
    ingestionForm.setValues({
      ...ingestionForm.values,
      file,
    });
  };

  const handleSelectCategory = (category: string) => {
    ingestionForm.setValues({
      ...ingestionForm.values,
      category_id: category,
    });
  };

  return (
    <div className="bg-primary-100 rounded-lg border-b border-gray-700 pb-6">
      <h2 className="text-lg text-gray-200 font-medium mb-4">
        Upload Local Document
      </h2>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          ingestionForm.handleSubmit(onSubmit);
        }}
        className="space-y-4"
      >
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

          <div className="xl:flex flex-col items-center sm:flex-row gap-2">
            <div className="flex-1">
              <FileInput
                ref={fileInputRef}
                label="Document File"
                onChange={handleSelectFile}
                filename={ingestionForm.values.file?.name}
                accept=".pdf,.doc,.docx,.txt,.csv,.xls,.xlsx,.zip"
                placeholder="Select a document"
              />
            </div>
          </div>

          <div className="xl:flex flex-col sm:flex-row space-y-2 sm:space-y-0 gap-2 items-end">
            <div className="flex-1">
              <TextInput
                label="Tags"
                name="tags"
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
              />
            </div>
            <Button
              type="button"
              onClick={handleAddNewTag}
              disabled={!newTag.trim()}
              className="bg-primary-500 text-gray-200 p-2 h-fit w-full sm:w-auto"
            >
              Add Tag
            </Button>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mb-2">
          {ingestionForm.values.tags.map((tag: string) => (
            <Chip
              key={tag}
              label={tag}
              removeTag={handleRemoveTag}
              color="blue"
            />
          ))}
        </div>

        <Button
          type="submit"
          disabled={isUploading || !isFormValid()}
          className="w-full sm:w-auto"
        >
          {isUploading ? "Uploading..." : "Upload Document"}
        </Button>
      </form>
    </div>
  );
};

export default LocalFileUpload;