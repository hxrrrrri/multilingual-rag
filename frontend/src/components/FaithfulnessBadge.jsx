export default function FaithfulnessBadge({ label, score }) {
  const cfg = {
    entailed:     { bg: "bg-green-900",  text: "text-green-300",  icon: "✓", label: "Verified" },
    neutral:      { bg: "bg-yellow-900", text: "text-yellow-300", icon: "~", label: "Uncertain" },
    contradiction:{ bg: "bg-red-900",    text: "text-red-300",    icon: "✗", label: "Low confidence" },
    no_context:   { bg: "bg-gray-800",   text: "text-gray-400",   icon: "?", label: "No context" },
  };
  const c = cfg[label] || cfg.no_context;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${c.bg} ${c.text}`}>
      {c.icon} {c.label} {score != null && `(${(score * 100).toFixed(0)}%)`}
    </span>
  );
}
