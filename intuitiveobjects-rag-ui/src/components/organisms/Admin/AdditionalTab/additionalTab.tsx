import React, { useState, useEffect } from "react";
import { useAppDispatch } from "@/store/hooks";
import { Button } from "@/components/atoms/Button/Button";
import { orgResetChromaApi, orgResetMongoApi, orgDeleteFileApi, orgGetFilesApi } from "@/services/adminApi";
import toast from "react-hot-toast";

const AdditionalTab = () => {
  const dispatch = useAppDispatch();
  const [loading, setLoading] = useState(false);
  const [fileId, setFileId] = useState("");
  const [files, setFiles] = useState<{ id: string; file_name: string }[]>([]);

  // Fetch file list on mount
  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const response = await dispatch(orgGetFilesApi()).unwrap();
        console.log("Files response:", response.data);
        if (response.success && Array.isArray(response.data)) {
          setFiles(response.data);
        }
      } catch (err) {
        toast.error("Failed to fetch files");
      }
    };
    fetchFiles();
  }, [dispatch]);

  const handleResetChroma = async () => {
    setLoading(true);
    try {
      const response = await dispatch(orgResetChromaApi()).unwrap();
      toast[response.success ? "success" : "error"](
        response.success ? "ChromaDB reset successfully" : "Failed to reset ChromaDB"
      );
    } catch (err: any) {
      toast.error("Error resetting ChromaDB: " + (err.message || "Unknown error"));
    } finally {
      setLoading(false);
    }
  };

  const handleResetMongo = async () => {
    setLoading(true);
    try {
      const response = await dispatch(orgResetMongoApi()).unwrap();
      toast[response.success ? "success" : "error"](
        response.success ? "MongoDB reset successfully" : "Failed to reset MongoDB"
      );
    } catch (err: any) {
      toast.error("Error resetting MongoDB: " + (err.message || "Unknown error"));
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteFile = async () => {
    if (!fileId) {
      toast.error("Please select a file");
      return;
    }
    setLoading(true);
    try {
      const response = await dispatch(orgDeleteFileApi(fileId)).unwrap();
      toast[response.success ? "success" : "error"](
        response.success ? "File deleted successfully" : "Failed to delete file"
      );
      // Remove deleted file from local state
      if (response.success) {
        setFiles(files.filter((file) => file.id !== fileId));
        setFileId("");
      }
    } catch (err: any) {
      toast.error("Error deleting file: " + (err.message || "Unknown error"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-4">
      <h2 className="text-xl font-semibold text-white">Additional Tools</h2>
      <div className="flex gap-4 flex-wrap items-center">
        <Button onClick={handleResetChroma} disabled={loading} className="bg-red-600 text-white">
          Reset ChromaDB
        </Button>
        <Button onClick={handleResetMongo} disabled={loading} className="bg-yellow-500 text-white">
          Reset MongoDB
        </Button>
        <select
          value={fileId}
          onChange={(e) => setFileId(e.target.value)}
          className="px-2 py-1 rounded"
        >
          <option value="">Select File</option>
          {files.map((file) => (
            <option key={file.id} value={file.id}>
              {file.file_name}
            </option>
          ))}
        </select>
        <Button onClick={handleDeleteFile} disabled={loading || !fileId} className="bg-blue-600 text-white">
          Delete File
        </Button>
      </div>
    </div>
  );
};

export default AdditionalTab;
