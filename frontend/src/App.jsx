import { Routes, Route, NavLink } from "react-router-dom";
import UploadPage  from "./pages/UploadPage";
import QueryPage   from "./pages/QueryPage";
import DocumentsPage from "./pages/DocumentsPage";

export default function App() {
  const navCls = ({ isActive }) =>
    `px-4 py-2 rounded-md text-sm font-medium transition-colors ${
      isActive ? "bg-blue-600 text-white" : "text-gray-300 hover:bg-gray-700 hover:text-white"
    }`;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* ── Nav ── */}
      <nav className="bg-gray-900 border-b border-gray-800 px-6 py-3 flex items-center gap-6">
        <span className="text-lg font-bold text-white mr-4">📄 DocIQ</span>
        <NavLink to="/"          className={navCls}>Upload</NavLink>
        <NavLink to="/documents" className={navCls}>Documents</NavLink>
        <NavLink to="/query"     className={navCls}>Query</NavLink>
      </nav>

      {/* ── Content ── */}
      <main className="max-w-5xl mx-auto px-6 py-8">
        <Routes>
          <Route path="/"          element={<UploadPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="/query"     element={<QueryPage />} />
        </Routes>
      </main>
    </div>
  );
}
