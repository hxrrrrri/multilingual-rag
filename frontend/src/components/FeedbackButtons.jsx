import { useState } from "react";
import { ThumbsUp, ThumbsDown } from "lucide-react";
import { submitFeedback } from "../utils/api";

export default function FeedbackButtons({ query, answer, context, docIds }) {
  const [voted, setVoted] = useState(null);

  const vote = async (thumbsUp) => {
    if (voted) return;
    setVoted(thumbsUp ? "up" : "down");
    try {
      await submitFeedback({ query, answer, context, doc_ids: docIds, thumbs_up: thumbsUp });
    } catch (e) {
      console.error("Feedback error:", e);
    }
  };

  if (voted) {
    return <p className="text-xs text-gray-500 mt-2">Thanks for your feedback!</p>;
  }

  return (
    <div className="flex items-center gap-3 mt-3">
      <span className="text-xs text-gray-500">Was this helpful?</span>
      <button onClick={() => vote(true)}
        className="flex items-center gap-1 text-xs text-green-400 hover:text-green-300 transition-colors">
        <ThumbsUp size={14} /> Yes
      </button>
      <button onClick={() => vote(false)}
        className="flex items-center gap-1 text-xs text-red-400 hover:text-red-300 transition-colors">
        <ThumbsDown size={14} /> No
      </button>
    </div>
  );
}
