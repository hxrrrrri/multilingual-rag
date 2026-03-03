import { useState, useCallback } from "react";
import { queryDocuments, submitFeedback, QueryResponse } from "../utils/api";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: QueryResponse;
  loading?: boolean;
  error?: string;
}

export function useQuery() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const ask = useCallback(async (query: string, documentIds?: string[]) => {
    const msgId = Date.now().toString();

    // Add user message + placeholder assistant message
    setMessages(prev => [
      ...prev,
      { id: msgId + "_u", role: "user", content: query },
      { id: msgId + "_a", role: "assistant", content: "", loading: true },
    ]);
    setIsLoading(true);

    try {
      const result = await queryDocuments(query, documentIds);
      setMessages(prev => prev.map(m =>
        m.id === msgId + "_a"
          ? { ...m, content: result.answer, response: result, loading: false }
          : m
      ));
      return result;
    } catch (e: any) {
      setMessages(prev => prev.map(m =>
        m.id === msgId + "_a"
          ? { ...m, content: "", loading: false, error: e.message ?? "Query failed" }
          : m
      ));
    } finally {
      setIsLoading(false);
    }
  }, []);

  const sendFeedback = useCallback(async (
    queryId: string, sentiment: "positive" | "negative"
  ) => {
    await submitFeedback(queryId, sentiment);
    setMessages(prev => prev.map(m =>
      m.response?.query_id === queryId
        ? { ...m, response: { ...m.response!, _feedback: sentiment } as any }
        : m
    ));
  }, []);

  const clear = useCallback(() => setMessages([]), []);

  return { messages, isLoading, ask, sendFeedback, clear };
}
