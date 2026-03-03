import { useState } from "react";
import { Send, Loader } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { queryDocuments } from "../utils/api";
import FaithfulnessBadge from "../components/FaithfulnessBadge";
import SourceCard from "../components/SourceCard";
import FeedbackButtons from "../components/FeedbackButtons";

const EXAMPLE_QUERIES = [
  "What are the main eligibility criteria?",
  "Summarise the key findings of this document.",
  "What is the deadline mentioned in the document?",
];

export default function QueryPage() {
  const [query,   setQuery]   = useState("");
  const [docId,   setDocId]   = useState("");
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState("");

  const handleQuery = async (q = query) => {
    if (!q.trim()) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const { data } = await queryDocuments(q.trim(), docId || null);
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || "Query failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleQuery(); } };

  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-2">Query Documents</h1>
      <p className="text-gray-400 text-sm mb-6">
        Ask questions in English, Hindi, or Malayalam. The system retrieves relevant passages and generates a grounded answer.
      </p>

      {/* Optional doc filter */}
      <input
        value={docId} onChange={e => setDocId(e.target.value)}
        placeholder="Filter by Document ID (optional)"
        className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-sm text-gray-300 placeholder-gray-600 mb-3 focus:outline-none focus:border-blue-500"
      />

      {/* Query input */}
      <div className="flex gap-2">
        <textarea
          value={query} onChange={e => setQuery(e.target.value)} onKeyDown={handleKey}
          placeholder="Ask a question about your documents…"
          rows={3}
          className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-gray-200 placeholder-gray-600 resize-none focus:outline-none focus:border-blue-500"
        />
        <button onClick={() => handleQuery()} disabled={loading || !query.trim()}
          className="px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg text-white transition-colors">
          {loading ? <Loader size={20} className="animate-spin" /> : <Send size={20} />}
        </button>
      </div>

      {/* Example queries */}
      <div className="flex flex-wrap gap-2 mt-3">
        {EXAMPLE_QUERIES.map(q => (
          <button key={q} onClick={() => { setQuery(q); handleQuery(q); }}
            className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-gray-200 px-3 py-1 rounded-full transition-colors">
            {q}
          </button>
        ))}
      </div>

      {/* Error */}
      {error && <p className="mt-4 text-red-400 text-sm bg-red-950 border border-red-800 rounded-lg px-4 py-3">{error}</p>}

      {/* Result */}
      {result && (
        <div className="mt-6 space-y-4">
          {/* Answer card */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <div className="flex items-center justify-between mb-3">
              <h2 className="font-semibold text-white">Answer</h2>
              <div className="flex items-center gap-2">
                {result.regenerated && (
                  <span className="text-xs text-yellow-400 bg-yellow-950 px-2 py-0.5 rounded-full">Regenerated</span>
                )}
                <FaithfulnessBadge label={result.faithfulness_label} score={result.faithfulness_score} />
              </div>
            </div>
            <div className="prose prose-invert prose-sm max-w-none text-gray-200">
              <ReactMarkdown>{result.answer}</ReactMarkdown>
            </div>
            <FeedbackButtons
              query={query}
              answer={result.answer}
              context={result.sources.map(s => s.text_preview).join("\n")}
              docIds={result.sources.map(s => s.doc_id).filter(Boolean)}
            />
          </div>

          {/* Sources */}
          {result.sources?.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-2">
                Sources ({result.sources.length} passages retrieved)
              </h3>
              <div className="grid gap-2">
                {result.sources.map((s, i) => <SourceCard key={i} source={s} index={i} />)}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
