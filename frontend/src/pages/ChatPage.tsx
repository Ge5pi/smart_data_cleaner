import { useState, useContext, useEffect, useRef } from 'react';
import { AppContext } from '../contexts/AppContext';
import { useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import { Send, Loader, BrainCircuit } from 'lucide-react';
import api from '../api';

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
};

const ChatPage = () => {
  const { fileId, sessionId, setSessionId, token } = useContext(AppContext)!;
  const navigate = useNavigate();
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [currentQuery, setCurrentQuery] = useState("");
  const [isReplying, setIsReplying] = useState(false);
  const [isSessionLoading, setIsSessionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [activeSessionFileId, setActiveSessionFileId] = useState<string | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!fileId) {
      alert("Пожалуйста, сначала выберите файл на главной странице.");
      navigate('/');
      return;
    }
    if (token && fileId && (fileId !== activeSessionFileId || !sessionId)) {
      setIsSessionLoading(true);
      setError(null);
      setChatHistory([]);

      const sessionFormData = new FormData();
      sessionFormData.append("file_id", fileId);

      api.post("/sessions/start", sessionFormData)
        .then(res => {
          const newSessionId = res.data.session_id;
          setSessionId(newSessionId);
          setActiveSessionFileId(fileId);
          setChatHistory([{ role: 'assistant', content: `Сессия для файла успешно начата. Я готов к анализу. Что бы вы хотели узнать?` }]);
        })
        .catch(err => {
            const message = err.response?.data?.detail || "Не удалось запустить сессию. Пожалуйста, попробуйте вернуться на главную страницу и выбрать файл заново.";
            setError(message);
            console.error("Ошибка старта сессии", err);
            setSessionId(null);
            setActiveSessionFileId(null);
        })
        .finally(() => setIsSessionLoading(false));
    }
  }, [fileId, token, sessionId, activeSessionFileId, navigate, setSessionId]);
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  const handleSendQuery = async () => {
    if (!currentQuery.trim() || !sessionId || isReplying || !token) return;

    const userMessage: ChatMessage = { role: 'user', content: currentQuery };
    setChatHistory(prev => [...prev, userMessage]);
    const queryToSend = currentQuery;
    setCurrentQuery("");
    setIsReplying(true);
    setError(null);

    const formData = new FormData();
    formData.append("session_id", sessionId);
    formData.append("query", queryToSend);

    try {
      const res = await api.post("/sessions/ask", formData);
      setChatHistory(prev => [...prev, { role: 'assistant', content: res.data.answer }]);
    } catch (err: any) {
        const message = err.response?.data?.detail || "Произошла ошибка при обработке вашего запроса.";
        setError(message);
        console.error(err);
    } finally {
      setIsReplying(false);
    }
  };

  if (isSessionLoading) {
    return <div className="flex justify-center items-center h-64 text-lg font-medium text-gray-600"><Loader className="animate-spin mr-4" /> Перезапускаем сессию для нового файла...</div>;
  }

  if (!fileId || !sessionId) {
      return <div className="flex justify-center items-center h-64 text-lg font-medium text-gray-600">Ожидание начала сессии...</div>;
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50">
            <div className="p-6 border-b border-gray-200/50 bg-gradient-to-r from-indigo-50 to-blue-50 flex justify-between items-center">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-gradient-to-r from-indigo-500 to-blue-500 rounded-xl"><BrainCircuit className="w-6 h-6 text-white" /></div>
                    <div>
                        <h2 className="text-xl font-semibold text-gray-800">Диалог с AI Агентом</h2>
                        <p className="text-sm text-gray-500 mt-1">Задавайте вопросы на естественном языке</p>
                    </div>
                </div>
                {sessionId && <span className="px-3 py-1 bg-gray-200 text-gray-600 text-xs font-mono rounded-full">Session: {sessionId.substring(0,8)}...</span>}
            </div>

            <div className="p-6 space-y-4 h-[600px] flex flex-col">
                <div className="flex-grow overflow-y-auto space-y-6 pr-4">
                  {chatHistory.map((msg, idx) => (
                    <div key={idx} className={`flex items-start gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                      {msg.role === 'assistant' && (<div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-blue-500 flex-shrink-0"></div>)}
                      <div className={`max-w-xl p-4 rounded-2xl prose prose-sm max-w-none ${msg.role === 'user' ? 'bg-blue-600 text-white rounded-br-none' : 'bg-gray-100 text-gray-800 rounded-bl-none'}`}>
                        <ReactMarkdown children={msg.content} />
                      </div>
                    </div>
                  ))}
                  {isReplying && (
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-blue-500 flex-shrink-0"></div>
                      <div className="p-4 bg-gray-100 rounded-2xl rounded-bl-none"><Loader className="w-5 h-5 text-gray-500 animate-spin" /></div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>
                <div className="flex-shrink-0 pt-4 border-t border-gray-200/80">
                  <div className="relative">
                    <input
                      type="text" value={currentQuery} onChange={(e) => setCurrentQuery(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleSendQuery()}
                      placeholder={sessionId ? "Например: 'покажи топ 5 строк по зарплате'" : "Ожидание начала сессии..."}
                      className="w-full pl-4 pr-12 py-3 rounded-xl border-2 border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all"
                      disabled={!sessionId || isReplying}
                    />
                    <button onClick={handleSendQuery} disabled={!sessionId || isReplying || !currentQuery.trim()} className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:bg-gray-400 transition-colors">
                      <Send className="w-5 h-5" />
                    </button>
                  </div>
                </div>
            </div>
            {error && <div className="p-4 border-t border-red-200 bg-red-50 text-red-700">{error}</div>}
        </div>
    </div>
  );
};

export default ChatPage;