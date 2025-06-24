import React, { useState } from "react";
import axios from "axios";

type ColumnAnalysis = {
  column: string;
  dtype: string;
  nulls: number;
  unique: number;
  sample_values: string[];
};

const App = () => {
  const [preview, setPreview] = useState<any[]>([]);
  const [columns, setColumns] = useState<ColumnAnalysis[]>([]);
  const [file, setFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const previewRes = await axios.post("http://localhost:5643/upload/", formData);
      setPreview(previewRes.data.preview);

      const analysisRes = await axios.post("http://localhost:5643/analyze/", formData);
      setColumns(analysisRes.data.columns);
    } catch (error) {
      console.error("Ошибка при загрузке файла", error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 px-8 py-6 font-sans">
      <div className="max-w-6xl mx-auto space-y-6">
        <h1 className="text-3xl font-bold text-gray-800">Smart Data Cleaner</h1>

        <div className="flex items-center gap-4">
          <input type="file" accept=".csv" onChange={handleFileChange} />
          <button
            onClick={handleUpload}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition"
          >
            Загрузить CSV
          </button>
        </div>

        {columns.length > 0 && (
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Анализ колонок</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm text-left border border-gray-200">
                <thead className="bg-gray-100 text-gray-700">
                  <tr>
                    <th className="px-4 py-2 border">Колонка</th>
                    <th className="px-4 py-2 border">Тип</th>
                    <th className="px-4 py-2 border">Nulls</th>
                    <th className="px-4 py-2 border">Уникальные</th>
                    <th className="px-4 py-2 border">Примеры</th>
                  </tr>
                </thead>
                <tbody>
                  {columns.map((col, idx) => (
                    <tr key={idx} className="border-t">
                      <td className="px-4 py-2 border">{col.column}</td>
                      <td className="px-4 py-2 border">{col.dtype}</td>
                      <td className="px-4 py-2 border">{col.nulls}</td>
                      <td className="px-4 py-2 border">{col.unique}</td>
                      <td className="px-4 py-2 border">{col.sample_values.join(", ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {preview.length > 0 && (
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Превью данных</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm text-left border border-gray-200">
                <thead className="bg-gray-100 text-gray-700">
                  <tr>
                    {Object.keys(preview[0]).map((key, idx) => (
                      <th key={idx} className="px-4 py-2 border">
                        {key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.map((row, idx) => (
                    <tr key={idx} className="border-t">
                      {Object.values(row).map((val, i) => (
                        <td key={i} className="px-4 py-2 border">
                          {String(val)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
