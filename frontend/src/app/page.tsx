"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Bot, User, Cpu, Info, ChevronRight, Activity, Search } from "lucide-react";
import axios from "axios";

interface Message {
  id: string;
  role: "user" | "ai";
  content: string;
  metadata?: {
    route: string;
    confidence: number;
    reasoning: string;
    stage2: boolean;
    latency: number;
    sources: string[];
  };
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: query,
    };

    setMessages((prev) => [...prev, userMsg]);
    setQuery("");
    setLoading(true);

    try {
      const response = await axios.post("http://localhost:8000/query", {
        query: userMsg.content,
      });

      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "ai",
        content: response.data.answer,
        metadata: {
          route: response.data.route_used,
          confidence: response.data.confidence,
          reasoning: response.data.router_reasoning,
          stage2: response.data.stage2_invoked,
          latency: response.data.latency_ms,
          sources: response.data.sources,
        },
      };

      setMessages((prev) => [...prev, aiMsg]);
    } catch (error) {
      console.error(error);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "ai",
        content: "Xin lỗi, đã có lỗi xảy ra khi kết nối tới máy chủ.",
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center p-4 md:p-8 relative">
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-emerald-500/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-500/10 blur-[120px] rounded-full" />
      </div>

      <header className="w-full max-w-5xl z-10 mb-8 flex flex-col md:flex-row justify-between items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold gradient-text">Legal QA Assistant</h1>
          <p className="text-gray-400 text-sm">Reasoning-Aware Adaptive Routing System</p>
        </div>
        <div className="flex gap-4">
          <div className="glass px-4 py-2 rounded-full flex items-center gap-2 text-xs font-medium">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            Backend Online
          </div>
          <div className="glass px-4 py-2 rounded-full flex items-center gap-2 text-xs font-medium">
            <Cpu className="w-3 h-3 text-blue-400" />
            Llama 3 Active
          </div>
        </div>
      </header>

      <div className="w-full max-w-5xl flex-1 flex flex-col lg:flex-row gap-6 z-10 overflow-hidden">
        {/* Chat Section */}
        <div className="flex-1 flex flex-col glass rounded-3xl overflow-hidden min-h-[500px]">
          <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-center p-8 space-y-4">
                <div className="w-16 h-16 bg-emerald-500/10 rounded-full flex items-center justify-center">
                  <Bot className="w-8 h-8 text-emerald-500" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold mb-2">Chào mừng bạn!</h2>
                  <p className="text-gray-400 max-w-md">
                    Tôi là trợ lý pháp luật thông minh. Tôi có thể giải đáp các thắc mắc về văn bản pháp luật Việt Nam.
                  </p>
                </div>
              </div>
            )}

            <AnimatePresence>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div className={`max-w-[85%] p-4 rounded-2xl ${msg.role === "user" ? "chat-bubble-user" : "chat-bubble-ai"}`}>
                    <div className="flex items-center gap-2 mb-2">
                      {msg.role === "user" ? (
                        <>
                          <span className="text-xs font-semibold text-blue-400">Bạn</span>
                          <User className="w-3 h-3 text-blue-400" />
                        </>
                      ) : (
                        <>
                          <Bot className="w-3 h-3 text-emerald-500" />
                          <span className="text-xs font-semibold text-emerald-500">AI Legal</span>
                        </>
                      )}
                    </div>
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                    
                    {msg.metadata && (
                      <div className="mt-4 pt-4 border-t border-white/10 flex flex-wrap gap-2">
                        <span className="text-[10px] bg-white/5 px-2 py-1 rounded flex items-center gap-1">
                          <Activity className="w-2 h-2" /> {msg.metadata.route}
                        </span>
                        <span className="text-[10px] bg-white/5 px-2 py-1 rounded">
                          {(msg.metadata.confidence * 100).toFixed(0)}% confidence
                        </span>
                        {msg.metadata.sources.length > 0 && (
                          <span className="text-[10px] bg-white/5 px-2 py-1 rounded">
                            {msg.metadata.sources.length} sources
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className="p-4 bg-white/5 border-t border-white/10">
            <div className="relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Nhập câu hỏi của bạn..."
                className="w-full bg-black/40 border border-white/10 rounded-2xl py-4 pl-4 pr-12 text-sm focus:outline-none focus:border-emerald-500/50 transition-colors"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="absolute right-2 top-2 bottom-2 w-10 flex items-center justify-center bg-emerald-500 rounded-xl hover:bg-emerald-600 disabled:opacity-50 disabled:hover:bg-emerald-500 transition-colors"
              >
                <Send className="w-4 h-4 text-black" />
              </button>
            </div>
          </form>
        </div>

        {/* Sidebar Info Section */}
        <div className="w-full lg:w-80 flex flex-col gap-6">
          <div className="glass p-6 rounded-3xl space-y-4">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-500" />
              Reasoning Stats
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-[10px] text-gray-400 mb-1">
                  <span>Routing Accuracy</span>
                  <span>100%</span>
                </div>
                <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 w-full" />
                </div>
              </div>
              
              <div className="grid grid-cols-3 gap-2 pt-2">
                <div className="bg-white/5 p-2 rounded-xl text-center">
                  <div className="text-sm font-bold">Vector</div>
                  <div className="text-[7px] uppercase text-gray-500">Fast</div>
                </div>
                <div className="bg-white/5 p-2 rounded-xl text-center">
                  <div className="text-sm font-bold">Graph</div>
                  <div className="text-[7px] uppercase text-gray-500">Reason</div>
                </div>
                <div className="bg-white/5 p-2 rounded-xl text-center">
                  <div className="text-sm font-bold text-amber-500">Clarify</div>
                  <div className="text-[7px] uppercase text-gray-500">Vague</div>
                </div>
              </div>
            </div>
          </div>

          <div className="glass p-6 rounded-3xl space-y-4 flex-1">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Info className="w-4 h-4 text-blue-400" />
              About System
            </h3>
            <div className="text-[11px] text-gray-400 space-y-3 leading-relaxed">
              <p>
                Hệ thống sử dụng mô hình <strong>Adaptive Routing</strong> hai giai đoạn để tối ưu hóa việc truy vấn.
              </p>
              <ul className="space-y-2">
                <li className="flex gap-2">
                  <ChevronRight className="w-3 h-3 text-emerald-500 shrink-0" />
                  <span><strong>Vector RAG:</strong> Dùng cho các câu hỏi tra cứu thông tin trực tiếp.</span>
                </li>
                <li className="flex gap-2">
                  <ChevronRight className="w-3 h-3 text-blue-500 shrink-0" />
                  <span><strong>Graph RAG:</strong> Dùng cho các câu hỏi cần suy luận đa chặng.</span>
                </li>
                <li className="flex gap-2">
                  <ChevronRight className="w-3 h-3 text-amber-500 shrink-0" />
                  <span><strong>Clarify:</strong> Tự động yêu cầu làm rõ nếu câu hỏi bị mơ hồ hoặc thiếu thực thể.</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
