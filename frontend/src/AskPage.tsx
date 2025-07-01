import React, { useState } from "react";
import axios from "axios";
import { Zap } from "lucide-react";

const AskPage = () => {
  const [fileId, setFileId] = useState<string>("");
  const [question, setQuestion] = useState<string>("");
  const [answer, setAnswer] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<{ question: string; answer: string }[]>([]);

  const handleAsk = async () => {
    if (!fileId || !question) return;

    setIsLoading(true);
    setError(null);
    setAnswer("");


    const formData = new FormData();
    formData.append("file_id", fileId);
    formData.append("question", question);

    try {
      const res = await axios.post("http://localhost:5643/ask/", formData);
      setAnswer(res.data.answer);
      setChatHistory((prev) => [...prev, { question, answer: res.data.answer }]);
    } catch (err) {
      console.error("Ошибка при запросе:", err);
      setError("Не удалось получить ответ. Попробуйте снова.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-6 py-12 space-y-6">
      <div className="flex flex-col space-y-4">
        <label className="text-sm font-semibold text-gray-700">
          ID загруженного файла:
        </label>
        <input
          type="text"
          className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={fileId}
          onChange={(e) => setFileId(e.target.value)}
          placeholder="Введите file_id, полученный после загрузки"
        />

        <label className="text-sm font-semibold text-gray-700">
          Введите вопрос о данных:
        </label>
        <textarea
          rows={4}
          className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Например: Какие столбцы содержат пропущенные значения?"
        />

        <button
          onClick={handleAsk}
          disabled={isLoading || !fileId || !question}
          className="flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-medium hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <>
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
              Отправка запроса...
            </>
          ) : (
            <>
              <Zap className="w-5 h-5" />
              Задать вопрос
            </>
          )}
        </button>

        {error && <p className="text-red-500 text-sm">{error}</p>}

        {answer && (
          <div className="p-4 mt-6 bg-white shadow-md rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-800 mb-2">Ответ:</h3>
            <p className="text-gray-700 whitespace-pre-line">{answer}</p>
          </div>
        )}
      </div>
      {chatHistory.length > 0 && (
  <div className="mt-10 space-y-6">
    <h3 className="text-xl font-semibold text-gray-800">История вопросов</h3>
    <div className="space-y-4">
      {chatHistory.map((entry, idx) => (
        <div key={idx} className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
          <p className="text-sm text-gray-500 mb-1">Вопрос:</p>
          <p className="text-gray-800 font-medium mb-2">{entry.question}</p>
          <p className="text-sm text-gray-500 mb-1">Ответ:</p>
          <p className="text-gray-700 whitespace-pre-line">{entry.answer}</p>
        </div>
      ))}
    </div>
  </div>
)}

    </div>
  );
};

export default AskPage;