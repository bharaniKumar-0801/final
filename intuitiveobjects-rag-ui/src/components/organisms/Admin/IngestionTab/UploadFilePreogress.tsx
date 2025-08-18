// import { FileIcon } from "@/components/atoms/Icon/FileIcon";

// const  UploadingFileProgress = (filename: string, size: string, progress: number) => {
// 	return (
// 		<div className="w-full max-w-[400px] bg-gray-800 rounded-lg p-4">
// 			<div className="mb-2 flex justify-between items-center">
// 				<div className="flex items-center gap-x-3">
// 					<span className="size-8 flex justify-center items-center border border-gray-200 text-gray-500 rounded-lg dark:border-neutral-700 dark:text-neutral-500">
// 						<FileIcon />
// 					</span>
// 					<div>
// 						<p className="text-sm font-medium text-gray-800 dark:text-white truncate max-w-[200px]">{filename}</p>
// 						<p className="text-xs text-gray-500 dark:text-neutral-500">{size}</p>
// 					</div>
// 				</div>
// 			</div>

// 			<div className="flex items-center gap-x-3 whitespace-nowrap">
// 				<div
// 					className="flex w-full h-2 bg-gray-200 rounded-full overflow-hidden dark:bg-neutral-700"
// 					role="progressbar"
// 					aria-valuenow={progress}
// 					aria-valuemin={0}
// 					aria-valuemax={100}>
// 					<div
// 						className="flex flex-col justify-center rounded-full overflow-hidden bg-blue-600 text-xs text-white text-center whitespace-nowrap transition duration-500 dark:bg-blue-500"
// 						style={{ width: `${progress}%` }}></div>
// 				</div>
// 				<div className="w-10 text-end">
// 					<span className="text-sm text-gray-800 dark:text-white">{progress}%</span>
// 				</div>
// 			</div>
// 		</div>
// 	);
// }

// export default UploadingFileProgress;


import { FileIcon } from "@/components/atoms/Icon/FileIcon";

const UploadingFileProgress = ({
  filename,
  size,
  progress,
  completed,
  total,
}: {
  filename: string;
  size: string;
  progress: number;
  completed?: number;
  total?: number;
}) => {
  return (
    <div className="w-full max-w-[420px] bg-[#0B1D39] rounded-md px-4 py-3 shadow-md text-white">
      <div className="mb-2 flex items-center gap-x-3">
        <div className="size-8 flex items-center justify-center border border-gray-500 rounded-md">
          <FileIcon />
        </div>
        <div className="overflow-hidden">
          <p className="text-sm font-semibold truncate">{filename}</p>
          <p className="text-xs text-gray-400">{size}</p>
          {typeof completed === "number" && typeof total === "number" && (
            <p className="text-xs text-gray-400">
              {completed} of {total} files processed
            </p>
          )}
        </div>
      </div>

      <div className="relative w-full h-2 rounded-full bg-gray-700 overflow-hidden">
        <div
          className="absolute h-full bg-blue-500 transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="text-center text-sm mt-2">{progress}% completed</div>
    </div>
  );
};

export default UploadingFileProgress;