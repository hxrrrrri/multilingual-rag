export default function SourceCard({ source, index }) {
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 text-sm">
      <div className="flex items-center justify-between mb-1">
        <span className="text-blue-400 font-medium">Source {index + 1}</span>
        <span className="text-gray-500 text-xs">Page {source.page}</span>
      </div>
      <p className="text-gray-300 line-clamp-3">{source.text_preview}</p>
      {source.score != null && (
        <div className="mt-1 text-xs text-gray-500">
          Relevance: {(source.score * 100).toFixed(1)}%
        </div>
      )}
    </div>
  );
}
