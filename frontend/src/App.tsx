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
      // Получаем превью
      const previewResponse = await axios.post("http://localhost:5643/upload/", formData);
      setPreview(previewResponse.data.preview);

      // Получаем анализ
      const analysisResponse = await axios.post("http://localhost:5643/analyze/", formData);
      setColumns(analysisResponse.data.columns);
    } catch (error) {
      console.error("Ошибка при загрузке файла", error);
    }
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h1>Smart Data Cleaner</h1>

      <input type="file" accept=".csv" onChange={handleFileChange} />
      <button onClick={handleUpload}>Загрузить</button>

      <h2>Анализ колонок</h2>
      <table border={1} cellPadding={5}>
        <thead>
          <tr>
            <th>Название</th>
            <th>Тип</th>
            <th>Nulls</th>
            <th>Уникальные</th>
            <th>Примеры</th>
          </tr>
        </thead>
        <tbody>
          {columns.map((col, idx) => (
            <tr key={idx}>
              <td>{col.column}</td>
              <td>{col.dtype}</td>
              <td>{col.nulls}</td>
              <td>{col.unique}</td>
              <td>{col.sample_values.join(", ")}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>Превью данных</h2>
      <table border={1} cellPadding={5}>
        <thead>
          <tr>
            {preview.length > 0 &&
              Object.keys(preview[0]).map((key, idx) => <th key={idx}>{key}</th>)}
          </tr>
        </thead>
        <tbody>
          {preview.map((row, idx) => (
            <tr key={idx}>
              {Object.values(row).map((value, i) => (
                <td key={i}>{value}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default App;
