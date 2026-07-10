"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Bot, User, Cpu, Info, ChevronRight, Activity, PlusCircle, MessageSquare, Trash2, Edit2, Menu, X, Lock, Unlock, LogOut, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";

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
    resolved_query?: string;
  };
}

interface Session {
  session_id: string;
  last_updated: number;
  turn_count: number;
  title: string;
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>("");
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Auth States
  const [user, setUser] = useState<{username: string, token: string} | null>(null);
  const [isLoginMode, setIsLoginMode] = useState(true);
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [isInitializing, setIsInitializing] = useState(true);
  const [streamingMsgId, setStreamingMsgId] = useState<string | null>(null);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingMessageContent, setEditingMessageContent] = useState<string>("");

  const loadingTexts = [
    "Đang phân tích độ mơ hồ của câu hỏi...",
    "Đang định tuyến (Vector/Graph)...",
    "Đang truy xuất CSDL pháp luật...",
    "Đang tổng hợp câu trả lời bằng LLM..."
  ];

  // Setup Auth and Axios Interceptors
  useEffect(() => {
    const token = localStorage.getItem("token");
    const storedUsername = localStorage.getItem("username");
    
    if (token && storedUsername) {
      setUser({ username: storedUsername, token });
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      fetchSessions();
      
      const savedSessionId = localStorage.getItem("currentSessionId");
      if (savedSessionId) {
        loadSession(savedSessionId);
      } else {
        startNewChat();
      }
    }
    setIsInitializing(false);
  }, []);

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError("");
    setLoading(true);
    
    try {
      const endpoint = isLoginMode ? "/auth/login" : "/auth/register";
      const res = await axios.post(`/api${endpoint}`, {
        username: authUsername,
        password: authPassword
      });
      
      if (isLoginMode) {
        const { access_token, username } = res.data;
        localStorage.setItem("token", access_token);
        localStorage.setItem("username", username);
        axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
        setUser({ username, token: access_token });
        fetchSessions();
        startNewChat();
      } else {
        setIsLoginMode(true);
        setAuthError("Đăng ký thành công! Vui lòng đăng nhập.");
      }
    } catch (err: any) {
      setAuthError(err.response?.data?.detail || "Đã có lỗi xảy ra");
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    localStorage.removeItem("currentSessionId");
    delete axios.defaults.headers.common['Authorization'];
    setUser(null);
    setSessions([]);
    setMessages([]);
  };

  const fetchSessions = async () => {
    try {
      const res = await axios.get("/api/sessions");
      setSessions(res.data.sessions);
    } catch (e) {
      console.error("Failed to fetch sessions", e);
    }
  };

  const startNewChat = () => {
    const newId = "session_" + Date.now().toString();
    setCurrentSessionId(newId);
    localStorage.setItem("currentSessionId", newId);
    setMessages([]);
  };

  const loadSession = async (sessionId: string) => {
    setCurrentSessionId(sessionId);
    localStorage.setItem("currentSessionId", sessionId);
    setMessages([]);
    try {
      const res = await axios.get(`/api/sessions/${sessionId}`);
      const history = res.data.history;
      const loadedMessages: Message[] = [];
      history.forEach((turn: any, index: number) => {
        loadedMessages.push({
          id: `u_${index}`,
          role: "user",
          content: turn.query
        });
        loadedMessages.push({
          id: `a_${index}`,
          role: "ai",
          content: turn.response,
          metadata: {
            route: turn.route_used,
            confidence: 1.0,
            reasoning: "",
            stage2: false,
            latency: 0,
            sources: []
          }
        });
      });
      setMessages(loadedMessages);
    } catch (e) {
      console.error("Failed to load session", e);
      startNewChat();
    }
  };

  const startRename = (session: Session, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingSessionId(session.session_id);
    setEditingTitle(session.title);
  };

  const submitRename = async (sessionId: string, e: React.FormEvent | React.FocusEvent) => {
    e.stopPropagation();
    e.preventDefault();
    if (!editingTitle.trim()) {
      setEditingSessionId(null);
      return;
    }
    try {
      await axios.put(`/api/sessions/${sessionId}`, { title: editingTitle });
      setSessions(prev => prev.map(s => s.session_id === sessionId ? { ...s, title: editingTitle } : s));
    } catch (err) {
      console.error("Failed to rename session", err);
    }
    setEditingSessionId(null);
  };

  const deleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await axios.delete(`/api/sessions/${sessionId}`);
      setSessions(prev => prev.filter(s => s.session_id !== sessionId));
      if (currentSessionId === sessionId) {
        startNewChat();
      }
    } catch (err) {
      console.error("Failed to delete session", err);
    }
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (loading) {
      setLoadingStep(0);
      interval = setInterval(() => {
        setLoadingStep((prev) => (prev < 3 ? prev + 1 : prev));
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [loading]);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const sendMessage = async (text: string) => {
    if (!text.trim() || loading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: text,
    };

    setMessages((prev) => [...prev, userMsg]);
    setQuery("");
    setLoading(true);

    try {
      const response = await fetch("/api/query_stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${user?.token}`
        },
        body: JSON.stringify({ query: userMsg.content, session_id: currentSessionId, verbose: true }),
      });

      if (!response.body) throw new Error("No response body");

      setLoading(false); // Hide loading bubble once we start receiving the stream

      const aiMsgId = (Date.now() + 1).toString();
      const aiMsg: Message = {
        id: aiMsgId,
        role: "ai",
        content: "",
      };
      setMessages((prev) => [...prev, aiMsg]);
      setStreamingMsgId(aiMsgId);

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let fullContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        
        const metadataIdx = chunk.indexOf('{"__metadata__": true');
        if (metadataIdx !== -1) {
          const textPart = chunk.substring(0, metadataIdx);
          const jsonPart = chunk.substring(metadataIdx);
          
          if (textPart) fullContent += textPart;
          
          try {
            const metadataStr = jsonPart.trim().split('\n')[0]; // Extract the JSON line
            const metadata = JSON.parse(metadataStr);
            setMessages((prev) => prev.map(msg => 
              msg.id === aiMsgId 
                ? { 
                    ...msg, 
                    content: fullContent,
                    metadata: {
                      route: metadata.route_used,
                      confidence: metadata.confidence,
                      reasoning: "",
                      stage2: metadata.stage2_invoked || false,
                      latency: metadata.latency_ms || 0,
                      sources: metadata.sources,
                      resolved_query: metadata.resolved_query,
                    }
                  } 
                : msg
            ));
          } catch (e) {
            console.error("Failed to parse metadata", e);
          }
        } else {
          fullContent += chunk;
          setMessages((prev) => prev.map(msg => 
            msg.id === aiMsgId ? { ...msg, content: fullContent } : msg
          ));
        }
      }
      
      // Refresh session list after sending a message to update sidebar
      fetchSessions();
      
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
      setStreamingMsgId(null);
    }
  };

  const submitEditMessage = async (msgIndex: number) => {
    if (!editingMessageContent.trim()) return;
    const turnIndex = messages.slice(0, msgIndex).filter(m => m.role === 'user').length;
    setEditingMessageId(null);
    setLoading(true);
    
    try {
      await axios.delete(`/api/sessions/${currentSessionId}/truncate?keep_turns=${turnIndex}`);
      setMessages(prev => prev.slice(0, msgIndex));
      sendMessage(editingMessageContent);
    } catch (e) {
      console.error("Failed to truncate session", e);
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(query);
  };

  if (isInitializing) return <div className="min-h-screen bg-[#0A0A0A]" />;

  if (!user) {
    return (
      <main className="min-h-screen flex items-center justify-center bg-[#0A0A0A] p-4 relative overflow-hidden text-white">
        <div className="absolute inset-0 z-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-[30%] h-[30%] bg-emerald-500/20 blur-[120px] rounded-full" />
          <div className="absolute bottom-1/4 right-1/4 w-[30%] h-[30%] bg-blue-500/20 blur-[120px] rounded-full" />
        </div>
        
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass p-8 md:p-12 rounded-3xl w-full max-w-md z-10 flex flex-col gap-6"
        >
          <div className="text-center space-y-2">
            <div className="w-16 h-16 bg-emerald-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
              <Bot className="w-8 h-8 text-emerald-500" />
            </div>
            <h1 className="text-3xl font-bold gradient-text">AI Legal</h1>
            <p className="text-gray-400 text-sm">Demo Dashboard Login</p>
          </div>

          <form onSubmit={handleAuth} className="space-y-4">
            <div>
              <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Username</label>
              <input 
                type="text"
                value={authUsername}
                onChange={(e) => setAuthUsername(e.target.value)}
                className="w-full mt-2 bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-emerald-500/50 transition-colors"
                placeholder="Ví dụ: hoidong_1"
                required
              />
            </div>
            <div>
              <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Password</label>
              <input 
                type="password"
                value={authPassword}
                onChange={(e) => setAuthPassword(e.target.value)}
                className="w-full mt-2 bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-emerald-500/50 transition-colors"
                placeholder="••••••••"
                required
              />
            </div>
            
            {authError && (
              <div className={`text-sm text-center ${authError.includes('thành công') ? 'text-emerald-400' : 'text-red-400'}`}>
                {authError}
              </div>
            )}

            <button 
              type="submit"
              disabled={loading}
              className="w-full bg-emerald-500 hover:bg-emerald-600 text-black font-bold py-3 rounded-xl transition-colors disabled:opacity-50 mt-4 flex items-center justify-center gap-2"
            >
              {loading ? <div className="w-5 h-5 border-2 border-black/20 border-t-black rounded-full animate-spin" /> : (isLoginMode ? <Unlock className="w-4 h-4" /> : <PlusCircle className="w-4 h-4" />)}
              {isLoginMode ? 'Đăng nhập' : 'Tạo tài khoản'}
            </button>
          </form>

          <div className="text-center mt-4">
            <button 
              onClick={() => { setIsLoginMode(!isLoginMode); setAuthError(""); }}
              className="text-sm text-emerald-400 hover:underline"
            >
              {isLoginMode ? 'Chưa có tài khoản? Đăng ký ngay' : 'Đã có tài khoản? Đăng nhập'}
            </button>
          </div>
        </motion.div>
      </main>
    );
  }

  return (
    <main className="h-screen w-full flex overflow-hidden bg-[#0A0A0A] text-white">
      {/* Background Glow */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-emerald-500/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-500/10 blur-[120px] rounded-full" />
      </div>

      {/* Mobile Sidebar Overlay */}
      {!isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-20 lg:hidden"
          onClick={() => setIsSidebarOpen(true)}
        />
      )}

      {/* Left Sidebar */}
      <div 
        className={`fixed lg:relative z-30 h-full bg-black/40 border-r border-white/10 flex flex-col transition-all duration-300 ease-in-out
          ${isSidebarOpen ? 'w-64 translate-x-0' : 'w-[68px] -translate-x-full lg:translate-x-0'}`}
      >
        <div className={`p-4 flex ${isSidebarOpen ? 'items-center justify-between' : 'flex-col items-center gap-4'} shrink-0`}>
          <button 
            onClick={startNewChat}
            className={`py-3 rounded-xl bg-emerald-500 hover:bg-emerald-600 text-black font-semibold flex items-center justify-center gap-2 transition-colors overflow-hidden shrink-0
              ${isSidebarOpen ? 'flex-1 px-4 mr-2' : 'w-9 h-9 mx-auto p-0 hidden lg:flex rounded-full'}`}
            title="New Chat"
          >
            <PlusCircle className="w-5 h-5 shrink-0" />
            {isSidebarOpen && <span className="whitespace-nowrap">New Chat</span>}
          </button>
          
          <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className={`p-2 hover:bg-white/10 rounded-xl hidden lg:block shrink-0`}
            title={isSidebarOpen ? "Đóng sidebar" : "Mở sidebar"}
          >
            {isSidebarOpen ? <PanelLeftClose className="w-5 h-5 text-gray-400" /> : <PanelLeftOpen className="w-5 h-5 text-gray-400" />}
          </button>

          {isSidebarOpen && (
            <button 
              onClick={() => setIsSidebarOpen(false)}
              className="p-2 hover:bg-white/10 rounded-xl lg:hidden shrink-0"
            >
              <X className="w-5 h-5 text-gray-400" />
            </button>
          )}
        </div>
        
        <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-1 px-3 pb-4">
          <h3 className={`text-[10px] font-bold text-gray-500 uppercase mb-2 px-2 mt-2 ${!isSidebarOpen && 'hidden'}`}>Lịch sử hôm nay</h3>
          {sessions.map(session => (
            <div 
              key={session.session_id}
              onClick={() => {
                if (editingSessionId !== session.session_id) loadSession(session.session_id);
              }}
              className={`group flex items-center p-2 rounded-lg cursor-pointer transition-colors ${currentSessionId === session.session_id ? 'bg-white/10' : 'hover:bg-white/5'} ${isSidebarOpen ? 'justify-between' : 'justify-center'}`}
              title={!isSidebarOpen ? session.title : undefined}
            >
              <div className={`flex items-center gap-3 overflow-hidden ${isSidebarOpen ? 'flex-1 min-w-0' : ''}`}>
                <MessageSquare className={`w-4 h-4 shrink-0 ${currentSessionId === session.session_id ? 'text-emerald-400' : 'text-gray-400'}`} />
                {isSidebarOpen && (
                  <div className="flex flex-col overflow-hidden flex-1 min-w-0">
                    {editingSessionId === session.session_id ? (
                      <input 
                        type="text" 
                        value={editingTitle} 
                        onChange={(e) => setEditingTitle(e.target.value)}
                        onBlur={(e) => submitRename(session.session_id, e)}
                        onKeyDown={(e) => { if (e.key === 'Enter') submitRename(session.session_id, e); }}
                        autoFocus
                        className="bg-black/50 text-sm text-white px-1 py-0.5 rounded outline-none border border-emerald-500/50 w-full"
                        onClick={(e) => e.stopPropagation()}
                      />
                    ) : (
                      <span className="text-sm text-gray-200 truncate block w-full">{session.title}</span>
                    )}
                    <span className="text-[10px] text-gray-500">{session.turn_count} turns</span>
                  </div>
                )}
              </div>
              {isSidebarOpen && (
                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity ml-2 shrink-0">
                  <button 
                    onClick={(e) => startRename(session, e)}
                    className="hover:text-emerald-400 p-1"
                  >
                    <Edit2 className="w-3.5 h-3.5 text-gray-500 hover:text-emerald-400" />
                  </button>
                  <button 
                    onClick={(e) => deleteSession(session.session_id, e)}
                    className="hover:text-red-400 p-1"
                  >
                    <Trash2 className="w-3.5 h-3.5 text-gray-500 hover:text-red-400" />
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
        
        {/* User Profile / Status */}
        <div className={`p-4 border-t border-white/10 flex items-center ${isSidebarOpen ? 'justify-between' : 'flex-col justify-center gap-4'} shrink-0`}>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center shrink-0" title={user.username}>
              <User className="w-4 h-4 text-emerald-500" />
            </div>
            {isSidebarOpen && (
              <div className="flex flex-col overflow-hidden min-w-0">
                <span className="text-sm font-medium truncate">{user.username}</span>
                <span className="text-[10px] text-emerald-500 flex items-center gap-1">
                  <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse shrink-0" />
                  Online
                </span>
              </div>
            )}
          </div>
          {isSidebarOpen ? (
            <button 
              onClick={handleLogout}
              className="p-2 hover:bg-red-500/10 hover:text-red-400 rounded-lg transition-colors text-gray-500 shrink-0"
              title="Đăng xuất"
            >
              <LogOut className="w-4 h-4" />
            </button>
          ) : (
            <button 
              onClick={handleLogout}
              className="p-2 hover:bg-red-500/10 hover:text-red-400 rounded-lg transition-colors text-gray-500 hidden lg:block shrink-0"
              title="Đăng xuất"
            >
              <LogOut className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col z-10 relative overflow-hidden">
        
        {/* Header */}
        <header className="h-14 border-b border-white/5 flex items-center px-4 justify-between bg-black/20 backdrop-blur-md shrink-0">
          <div className="flex items-center gap-3">
            <button 
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 hover:bg-white/10 rounded-xl transition-colors lg:hidden"
              title="Mở sidebar"
            >
              <Menu className="w-5 h-5 text-gray-400" />
            </button>
            <h1 className="text-lg font-bold gradient-text">Legal QA Assistant</h1>
          </div>
          
          <div className="flex gap-3 items-center">
            <div className="hidden md:flex bg-white/5 px-3 py-1.5 rounded-full items-center gap-2 text-xs font-medium border border-white/5">
              <Cpu className="w-3 h-3 text-blue-400" />
              Llama 3
            </div>
            <div className="hidden md:flex bg-white/5 px-3 py-1.5 rounded-full items-center gap-2 text-xs font-medium border border-white/5">
              <Activity className="w-3 h-3 text-emerald-400" />
              Hybrid Routing
            </div>
          </div>
        </header>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 custom-scrollbar">
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-center py-20 space-y-6">
                <div className="w-20 h-20 bg-emerald-500/10 rounded-full flex items-center justify-center">
                  <Bot className="w-10 h-10 text-emerald-500" />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold mb-3">Tôi có thể giúp gì cho bạn hôm nay?</h2>
                  <p className="text-gray-400 max-w-md mx-auto text-sm">
                    Tôi là trợ lý AI am hiểu Pháp luật Việt Nam. Hãy đặt câu hỏi và tôi sẽ tìm kiếm, suy luận từ cơ sở dữ liệu để đưa ra câu trả lời chính xác nhất.
                  </p>
                </div>
              </div>
            )}

            <AnimatePresence>
              {messages.map((msg, msgIndex) => {
                const contentLines = msg.content.split('\n');
                const options = contentLines.filter(line => line.trim().startsWith('- [Option] '));
                const textContent = contentLines.filter(line => !line.trim().startsWith('- [Option] ')).join('\n');

                return (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} group/msg`}
                >
                  <div className={`max-w-[85%] p-4 rounded-2xl relative ${msg.role === "user" ? "chat-bubble-user" : "chat-bubble-ai"}`}>
                    
                    {msg.role === "user" && msg.id !== editingMessageId && !loading && (
                      <button 
                        onClick={() => {
                          setEditingMessageId(msg.id);
                          setEditingMessageContent(msg.content);
                        }}
                        className="absolute top-2 right-[100%] mr-2 p-2 rounded-full bg-white/5 hover:bg-white/10 text-gray-400 opacity-0 group-hover/msg:opacity-100 transition-opacity"
                        title="Sửa tin nhắn"
                      >
                        <Edit2 className="w-4 h-4" />
                      </button>
                    )}
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
                    {msg.role === "ai" && msg.metadata?.resolved_query && msgIndex > 0 && messages[msgIndex - 1]?.content !== msg.metadata.resolved_query && (
                      <div className="mb-3 text-xs text-emerald-400/80 bg-emerald-500/10 p-2.5 rounded-lg border border-emerald-500/20 italic">
                        <span className="font-semibold not-italic">Đã hiểu ngữ cảnh:</span> {msg.metadata.resolved_query}
                      </div>
                    )}
                    <div className="text-sm leading-relaxed prose prose-invert max-w-none">
                      {msg.id === editingMessageId ? (
                        <div className="flex flex-col gap-2">
                          <textarea 
                            value={editingMessageContent}
                            onChange={(e) => setEditingMessageContent(e.target.value)}
                            className="w-full bg-[#3F3F3F] text-white border-none rounded-xl p-3 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500 resize-none"
                            rows={3}
                            autoFocus
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                submitEditMessage(msgIndex);
                              } else if (e.key === 'Escape') {
                                setEditingMessageId(null);
                              }
                            }}
                          />
                          <div className="flex justify-end gap-2 mt-1">
                            <button 
                              onClick={() => setEditingMessageId(null)}
                              className="px-3 py-1.5 rounded-lg bg-gray-600 hover:bg-gray-500 text-xs font-semibold transition-colors"
                            >
                              Hủy
                            </button>
                            <button 
                              onClick={() => submitEditMessage(msgIndex)}
                              className="px-3 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-black text-xs font-semibold transition-colors"
                            >
                              Lưu & Gửi lại
                            </button>
                          </div>
                        </div>
                      ) : (
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm, remarkBreaks]}
                          components={{
                            p: ({ node, ...props }) => <p className="mb-3 last:mb-0" {...props} />,
                            strong: ({ node, ...props }) => <strong className="font-bold text-white" {...props} />,
                            em: ({ node, ...props }) => <em className="italic text-gray-300" {...props} />,
                            ul: ({ node, ...props }) => <ul className="list-disc ml-5 mb-3 space-y-1" {...props} />,
                            ol: ({ node, ...props }) => <ol className="list-decimal ml-5 mb-3 space-y-1" {...props} />,
                            li: ({ node, ...props }) => <li className="pl-1" {...props} />
                          }}
                        >
                          {textContent + (streamingMsgId === msg.id ? ' ▌' : '')}
                        </ReactMarkdown>
                      )}
                    </div>

                    {options.length > 0 && (
                      <div className="mt-4 flex flex-col gap-2">
                        {options.map((opt, idx) => {
                          const optionText = opt.replace('- [Option] ', '').trim();
                          return (
                            <button
                              key={idx}
                              onClick={() => sendMessage(optionText)}
                              disabled={loading}
                              className="text-left px-4 py-3 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/20 rounded-xl text-emerald-400 text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                              {optionText}
                            </button>
                          );
                        })}
                      </div>
                    )}
                    
                    {msg.metadata && msg.metadata.route && (
                      <div className="mt-4 pt-4 border-t border-white/10 flex flex-wrap gap-2">
                        <span className="text-[10px] bg-white/5 px-2 py-1 rounded flex items-center gap-1 border border-white/5">
                          <Activity className="w-2 h-2" /> {msg.metadata.route}
                        </span>
                        <span className="text-[10px] bg-white/5 px-2 py-1 rounded border border-white/5">
                          {(msg.metadata.confidence * 100).toFixed(0)}% conf
                        </span>
                        <span className="text-[10px] bg-white/5 px-2 py-1 rounded border border-white/5 text-emerald-400">
                          {msg.metadata.stage2 ? 'Stage 2 (LLM)' : 'Stage 1 (XGB)'}
                        </span>
                        {msg.metadata.latency > 0 && (
                          <span className="text-[10px] bg-white/5 px-2 py-1 rounded border border-white/5 text-blue-400">
                            {(msg.metadata.latency / 1000).toFixed(2)}s
                          </span>
                        )}
                        {msg.metadata.sources && msg.metadata.sources.length > 0 && (
                          <span className="text-[10px] bg-white/5 px-2 py-1 rounded border border-white/5">
                            {msg.metadata.sources.length} sources
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </motion.div>
              )})}
            </AnimatePresence>
            
            {loading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-start"
              >
                <div className="max-w-[85%] p-4 rounded-2xl chat-bubble-ai flex flex-col gap-3">
                  <div className="flex items-center gap-2">
                    <Bot className="w-3 h-3 text-emerald-500" />
                    <span className="text-xs font-semibold text-emerald-500">AI Legal</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex gap-1">
                      <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                      <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                      <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                    </div>
                    <span className="text-sm text-gray-300 animate-pulse">
                      {loadingTexts[loadingStep]}
                    </span>
                  </div>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="p-4 bg-gradient-to-t from-[#0A0A0A] via-[#0A0A0A] to-transparent pt-8">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSubmit} className="relative group">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Nhắn tin cho AI Legal..."
                className="w-full bg-[#2F2F2F] hover:bg-[#3F3F3F] focus:bg-[#3F3F3F] text-white border-none rounded-2xl py-4 pl-5 pr-14 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500/50 transition-all shadow-lg"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="absolute right-2 top-2 bottom-2 w-10 flex items-center justify-center bg-white text-black rounded-xl hover:bg-gray-200 disabled:opacity-50 disabled:hover:bg-white transition-colors"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
            <p className="text-[10px] text-center text-gray-500 mt-3">
              AI có thể mắc lỗi. Vui lòng kiểm tra lại các thông tin pháp lý quan trọng.
            </p>
          </div>
        </div>

      </div>
    </main>
  );
}
