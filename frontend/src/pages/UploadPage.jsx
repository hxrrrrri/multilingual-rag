import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileText, CheckCircle, AlertCircle, Loader } from "lucide-react";
import { uploadDocument } from "../utils/api";

export default function UploadPage() {
  const [file,     setFile]     = useState(null);
  const [progress, setProgress] = useState(0);
  const [status,   setStatus]   = useState("idle"); // idle | uploading | success | error
  const [result,   setResult]   = useState(null);
  const [error,    setError]    = useState("");

  const onDrop = useCallback((accepted) => {
    if (accepted.length) { setFile(accepted[0]); setStatus("idle"); setError(""); }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxFiles: 1,
  });

  const handleUpload = async () => {
    if (!file) return;
    setStatus("uploading"); setProgress(0); setError("");
    try {
      const { data } = await uploadDocument(file, setProgress);
      setResult(data); setStatus("success");
    } catch (e) {
      setError(e.response?.data?.detail || "Upload failed. Is the backend running?");
      setStatus("error");
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-2">Upload Document</h1>
      <p className="text-gray-400 mb-6 text-sm">
        Upload a PDF in English, Hindi, or Malayalam. The system will OCR, chunk, and index it automatically.
      </p>

      {/* Dropzone */}
      <div {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors
          ${isDragActive ? "border-blue-500 bg-blue-950" : "border-gray-700 hover:border-gray-500 bg-gray-900"}`}>
        <input {...getInputProps()} />
        <Upload className="mx-auto mb-3 text-gray-500" size={36} />
        {file ? (
          <p className="text-gray-200 font-medium flex items-center justify-center gap-2">
            <FileText size={18} className="text-blue-400" /> {file.name}
            <span className="text-gray-500 text-sm">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
          </p>
        ) : (
          <p className="text-gray-400">Drag &amp; drop a PDF here, or click to select</p>
        )}
      </div>

      {/* Upload Button */}
      <button onClick={handleUpload} disabled={!file || status === "uploading"}
        className="mt-4 w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed
          text-white font-semibold rounded-lg transition-colors flex items-center justify-center gap-2">
        {status === "uploading" ? <><Loader size={16} className="animate-spin" /> Processing… ({progress}%)</> : "Upload & Index"}
      </button>

      {/* Progress bar */}
      {status === "uploading" && (
        <div className="mt-3 bg-gray-800 rounded-full h-2">
          <div className="bg-blue-600 h-2 rounded-full transition-all" style={{ width: `${progress}%` }} />
        </div>
      )}

      {/* Success */}
      {status === "success" && result && (
        <div className="mt-4 bg-green-950 border border-green-800 rounded-lg p-4">
          <p className="flex items-center gap-2 text-green-400 font-medium">
            <CheckCircle size={18} /> Document indexed successfully!
          </p>
          <div className="mt-2 text-sm text-gray-300 grid grid-cols-2 gap-1">
            <span className="text-gray-500">Pages:</span>     <span>{result.page_count}</span>
            <span className="text-gray-500">Chunks:</span>    <span>{result.chunk_count}</span>
            <span className="text-gray-500">Language:</span>  <span className="uppercase">{result.language}</span>
            <span className="text-gray-500">Doc ID:</span>    <span className="font-mono text-xs">{result.id}</span>
          </div>
        </div>
      )}

      {/* Error */}
      {status === "error" && (
        <div className="mt-4 bg-red-950 border border-red-800 rounded-lg p-4 flex items-center gap-2 text-red-400">
          <AlertCircle size={18} /> {error}
        </div>
      )}
    </div>
  );
}
