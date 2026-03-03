import { useEffect, useState } from "react";
import { Trash2, FileText, RefreshCw, Loader } from "lucide-react";
import { listDocuments, deleteDocument } from "../utils/api";

const STATUS_COLOR = {
  completed:  "text-green-400",
  processing: "text-yellow-400",
  failed:     "text-red-400",
  pending:    "text-gray-400",
};

export default function DocumentsPage() {
  const [docs,    setDocs]    = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState("");

  const load = async () => {
    setLoading(true);
    try {
      const { data } = await listDocuments();
      setDocs(data);
    } catch (e) {
      setError("Failed to load documents.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const handleDelete = async (id) => {
    if (!window.confirm("Delete this document and all its vectors?")) return;
    try {
      await deleteDocument(id);
      setDocs(prev => prev.filter(d => d.id !== id));
    } catch (e) {
      alert("Delete failed: " + (e.response?.data?.detail || e.message));
    }
  };

  if (loading) return (
    <div className="flex items-center justify-center py-16 text-gray-400">
      <Loader className="animate-spin mr-2" size={20} /> Loading documents…
    </div>
  );

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white">Documents</h1>
        <button onClick={load}
          className="flex items-center gap-2 text-sm text-gray-400 hover:text-white px-3 py-1.5 rounded-md bg-gray-800 hover:bg-gray-700">
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {error && <p className="text-red-400 mb-4">{error}</p>}

      {docs.length === 0 ? (
        <div className="text-center py-16 text-gray-500">
          <FileText size={48} className="mx-auto mb-3 opacity-30" />
          <p>No documents yet. Upload one to get started.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {docs.map(doc => (
            <div key={doc.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <FileText size={20} className="text-blue-400 shrink-0" />
                <div>
                  <p className="text-white font-medium text-sm">{doc.filename}</p>
                  <p className="text-gray-500 text-xs mt-0.5">
                    {doc.page_count} pages · {doc.chunk_count} chunks ·{" "}
                    <span className="uppercase">{doc.language}</span> ·{" "}
                    <span className={STATUS_COLOR[doc.status] || "text-gray-400"}>{doc.status}</span>
                  </p>
                </div>
              </div>
              <button onClick={() => handleDelete(doc.id)}
                className="text-gray-600 hover:text-red-400 transition-colors p-1">
                <Trash2 size={16} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
