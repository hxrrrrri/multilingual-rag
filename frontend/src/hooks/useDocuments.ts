import { useState, useEffect, useCallback } from "react";
import { listDocuments, getDocument, deleteDocument, uploadDocument, Document } from "../utils/api";

export function useDocuments() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const { documents: docs } = await listDocuments();
      setDocuments(docs);
    } catch (e: any) {
      setError(e.message ?? "Failed to load documents");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetch(); }, [fetch]);

  const upload = useCallback(async (file: File) => {
    const doc = await uploadDocument(file);
    setDocuments(prev => [doc, ...prev]);
    // Poll until ready
    const poll = setInterval(async () => {
      try {
        const updated = await getDocument(doc.document_id);
        setDocuments(prev => prev.map(d => d.document_id === updated.document_id ? updated : d));
        if (updated.status === "ready" || updated.status === "failed") {
          clearInterval(poll);
        }
      } catch { clearInterval(poll); }
    }, 2000);
    return doc;
  }, []);

  const remove = useCallback(async (id: string) => {
    await deleteDocument(id);
    setDocuments(prev => prev.filter(d => d.document_id !== id));
  }, []);

  return { documents, loading, error, fetch, upload, remove };
}
