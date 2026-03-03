"use client";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud, FileText, Loader2, CheckCircle, XCircle } from "lucide-react";
import clsx from "clsx";

interface Props {
  onUpload: (file: File) => Promise<any>;
}

interface FileState {
  file: File;
  status: "uploading" | "done" | "error";
  error?: string;
}

export default function UploadZone({ onUpload }: Props) {
  const [files, setFiles] = useState<FileState[]>([]);

  const onDrop = useCallback(async (accepted: File[]) => {
    for (const file of accepted) {
      setFiles(prev => [...prev, { file, status: "uploading" }]);
      try {
        await onUpload(file);
        setFiles(prev => prev.map(f =>
          f.file.name === file.name ? { ...f, status: "done" } : f
        ));
      } catch (e: any) {
        setFiles(prev => prev.map(f =>
          f.file.name === file.name ? { ...f, status: "error", error: e.message } : f
        ));
      }
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"], "image/*": [".png", ".jpg", ".jpeg", ".tiff"] },
    maxSize: 50 * 1024 * 1024,
  });

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={clsx(
          "border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors",
          isDragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"
        )}
      >
        <input {...getInputProps()} />
        <UploadCloud className="mx-auto mb-3 text-gray-400" size={40} />
        <p className="text-base font-medium text-gray-700">
          {isDragActive ? "Drop files here..." : "Drag & drop or click to upload"}
        </p>
        <p className="text-sm text-gray-400 mt-1">PDF, PNG, JPG, TIFF — up to 50 MB</p>
        <p className="text-xs text-gray-400 mt-1">Supports English, Hindi, Malayalam</p>
      </div>

      {files.length > 0 && (
        <ul className="space-y-2">
          {files.map((f, i) => (
            <li key={i} className="flex items-center gap-3 p-3 rounded-lg border bg-white text-sm">
              <FileText size={16} className="text-gray-400 shrink-0" />
              <span className="flex-1 truncate text-gray-700">{f.file.name}</span>
              <span className="text-xs text-gray-400">{(f.file.size / 1024).toFixed(0)} KB</span>
              {f.status === "uploading" && <Loader2 size={16} className="animate-spin text-blue-500 shrink-0" />}
              {f.status === "done" && <CheckCircle size={16} className="text-green-500 shrink-0" />}
              {f.status === "error" && (
                <span className="flex items-center gap-1 text-red-500">
                  <XCircle size={16} />
                  <span className="text-xs">{f.error}</span>
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
