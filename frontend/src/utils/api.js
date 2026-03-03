import axios from "axios";

const api = axios.create({ baseURL: "/api/v1" });

export const uploadDocument = (file, onProgress) => {
  const fd = new FormData();
  fd.append("file", file);
  return api.post("/documents/ingest", fd, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => onProgress && onProgress(Math.round((e.loaded / e.total) * 100)),
  });
};

export const listDocuments = () => api.get("/documents/");
export const deleteDocument = (id) => api.delete(`/documents/${id}`);
export const queryDocuments = (query, docId = null) =>
  api.post("/query", { query, doc_id: docId });
export const submitFeedback = (payload) => api.post("/feedback/", payload);
export const getHealth = () => api.get("/health");
