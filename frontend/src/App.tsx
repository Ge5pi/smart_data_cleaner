import React, { useState } from "react";
import { Upload, FileText, AlertTriangle, BarChart3, Database, CheckCircle2, Eye, Filter, Zap, TrendingUp, ArrowDownCircle } from "lucide-react";

import axios from "axios";

type ColumnAnalysis = {
  column: string;
  dtype: string;
  nulls: number;
  unique: number;
  sample_values: string[];
};

type ImputationResult = {
  preview: any[];
  missing_before: { [key: string]: number };
  missing_after: { [key: string]: number };
  processing_results: { [key: string]: string };
  total_rows: number;
};

const App = () => {
  // State variables for the application
  const [outliers, setOutliers] = useState<any[]>([]);
  const [outlierCount, setOutlierCount] = useState<number | null>(null);
  const [preview, setPreview] = useState<any[]>([]);
  const [columns, setColumns] = useState<ColumnAnalysis[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [selectedMissingColumns, setSelectedMissingColumns] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isImputing, setIsImputing] = useState(false);
  const [imputationResult, setImputationResult] = useState<ImputationResult | null>(null);

  // Handles file selection from the input
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0]);
      // Reset states when a new file is selected
      setColumns([]);
      setPreview([]);
      setOutliers([]);
      setOutlierCount(null);
      setSelectedColumns([]);
      setSelectedMissingColumns([]);
      setImputationResult(null);
    }
  };

  const handleDownloadCleaned = async () => {
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);
  formData.append("outlier_columns", JSON.stringify(selectedColumns));
  formData.append("impute_columns", JSON.stringify(selectedMissingColumns));

  try {
    const res = await axios.post("http://localhost:5643/download-cleaned/", formData, {
      responseType: "blob",
    });

    const blob = new Blob([res.data], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cleaned_data.csv";
    a.click();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    console.error("Ошибка при скачивании очищенного файла", err);
  }
};

  // Handles the file upload and initial data analysis
  const handleUpload = async () => {
    if (!file) return;
    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Send the file for preview
      const previewRes = await axios.post("http://localhost:5643/upload/", formData);
      setPreview(previewRes.data.preview);

      // Send the file for column analysis
      const analysisRes = await axios.post("http://localhost:5643/analyze/", formData);
      setColumns(analysisRes.data.columns);
    } catch (error) {
      console.error("Ошибка при загрузке файла:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Handles missing values imputation with TabPFN
  const handleImputeMissing = async () => {
    if (!file || selectedMissingColumns.length === 0) return;
    setIsImputing(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("columns", JSON.stringify(selectedMissingColumns));

    try {
      const res = await axios.post("http://localhost:5643/impute-missing/", formData);
      setImputationResult(res.data);
    } catch (error) {
      console.error("Ошибка при заполнении пропусков:", error);
    } finally {
      setIsImputing(false);
    }
  };

  // Handles the outlier detection process
  const handleDetectOutliers = async () => {
    if (!file || selectedColumns.length === 0) return;
    setIsAnalyzing(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("columns", JSON.stringify(selectedColumns));

    try {
      const res = await axios.post("http://localhost:5643/outliers/", formData);
      setOutliers(res.data.outlier_preview);
      setOutlierCount(res.data.outlier_count);
    } catch (error) {
      console.error("Ошибка при определении выбросов:", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Determines the background and text color for data type tags
  const getTypeColor = (dtype: string) => {
    switch (dtype) {
      case "int64":
      case "float64":
        return "bg-blue-100 text-blue-800";
      case "object":
        return "bg-green-100 text-green-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  // Returns the appropriate icon for each data type
  const getTypeIcon = (dtype: string) => {
    switch (dtype) {
      case "int64":
      case "float64":
        return <BarChart3 className="w-4 h-4" />;
      case "object":
        return <FileText className="w-4 h-4" />;
      default:
        return <Database className="w-4 h-4" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 font-sans">
      {/* Header Section */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200/50 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl">
              <Database className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
              Smart Data Cleaner
            </h1>
            <div className="ml-auto flex items-center gap-2 text-sm text-gray-500">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              Готов к работе
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Upload Section */}
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-8">
          <div className="flex items-center gap-4 mb-6">
            <div className="p-3 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-xl">
              <Upload className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-800">Загрузка данных</h2>
              <p className="text-gray-600">Выберите CSV файл для анализа</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="relative">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-gray-100 to-gray-200 hover:from-gray-200 hover:to-gray-300 rounded-xl cursor-pointer transition-all duration-200 hover:shadow-md"
              >
                <FileText className="w-5 h-5 text-gray-600" />
                <span className="text-gray-700 font-medium">
                  {file ? file.name : "Выбрать файл"}
                </span>
              </label>
            </div>

            <button
              onClick={handleUpload}
              disabled={!file || isLoading}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 text-white rounded-xl font-medium transition-all duration-200 hover:shadow-lg transform hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  Загружаем...
                </>
              ) : (
                <>
                  <Upload className="w-5 h-5" />
                  Загрузить CSV
                </>
              )}
            </button>
          </div>
        </div>

        {/* Column Analysis Section */}
        {columns.length > 0 && (
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 overflow-hidden">
            <div className="p-6 border-b border-gray-200/50 bg-gradient-to-r from-green-50 to-emerald-50">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl">
                  <BarChart3 className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-gray-800">Анализ столбцов</h2>
                  <p className="text-gray-600">Обзор структуры данных</p>
                </div>
              </div>
            </div>

            <div className="overflow-x-auto max-h-[350px] overflow-auto">
              <table className="w-full ">
                <thead className="bg-gray-50/80">
                  <tr>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Столбец</th>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Тип данных</th>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Пустые значения</th>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Уникальные</th>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Примеры значений</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200/50">
                  {columns.map((col, idx) => (
                    <tr key={idx} className="hover:bg-gray-50/50 transition-colors duration-150">
                      <td className="px-6 py-4">
                        <div className="font-medium text-gray-900">{col.column}</div>
                      </td>
                      <td className="px-6 py-4">
                        <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium ${getTypeColor(col.dtype)}`}>
                          {getTypeIcon(col.dtype)}
                          {col.dtype}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          {col.nulls > 0 ? (
                            <AlertTriangle className="w-4 h-4 text-amber-500" />
                          ) : (
                            <CheckCircle2 className="w-4 h-4 text-green-500" />
                          )}
                          <span className={col.nulls > 0 ? "text-amber-700 font-medium" : "text-green-700"}>
                            {col.nulls}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-gray-900 font-medium">{col.unique}</td>
                      <td className="px-6 py-4">
                        <div className="flex flex-wrap gap-1">
                          {col.sample_values.slice(0, 3).map((val, i) => (
                            <span key={i} className="px-2 py-1 bg-gray-100 text-gray-700 rounded-md text-sm">
                              {val}
                            </span>
                          ))}
                          {col.sample_values.length > 3 && (
                            <span className="px-2 py-1 text-gray-500 text-sm">+{col.sample_values.length - 3}</span>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Missing Values Imputation Section */}
        {columns.length > 0 && columns.some(col => col.nulls > 0) && (
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-800 ">Заполнение пропусков с TabPFN</h2>
                <p className="text-gray-600">Выберите столбцы для интеллектуального заполнения пропущенных значений</p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-h-[350px] overflow-auto">
                {columns
                  .filter((col) => col.nulls > 0)
                  .map((col) => (
                    <label key={col.column} className="relative">
                      <input
                        type="checkbox"
                        value={col.column}
                        checked={selectedMissingColumns.includes(col.column)}
                        onChange={(e) => {
                          const val = e.target.value;
                          setSelectedMissingColumns((prev) =>
                            prev.includes(val)
                              ? prev.filter((c) => c !== val)
                              : [...prev, val]
                          );
                        }}
                        className="sr-only"
                      />
                      <div className={`p-4 rounded-xl border-2 cursor-pointer transition-all duration-200 ${
                        selectedMissingColumns.includes(col.column)
                          ? "border-purple-500 bg-purple-50 shadow-md"
                          : "border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm"
                      }`}>
                        <div className="flex items-center gap-3">
                          <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                            selectedMissingColumns.includes(col.column)
                              ? "border-purple-500 bg-purple-500"
                              : "border-gray-300"
                          }`}>
                            {selectedMissingColumns.includes(col.column) && (
                              <CheckCircle2 className="w-3 h-3 text-white" />
                            )}
                          </div>
                          <span className="font-medium text-gray-900">{col.column}</span>
                        </div>
                        <div className="mt-2 text-sm text-amber-600 font-medium">
                          {col.nulls} пропущенных значений
                        </div>
                      </div>
                    </label>
                  ))}
              </div>

              <button
                    onClick={handleDownloadCleaned}
                    className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 disabled:from-gray-400 disabled:to-gray-500 text-white rounded-xl font-medium transition-all duration-200 hover:shadow-lg transform hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed"
              >


                {isImputing ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    Заполняем пропуски...
                  </>
                ) : (
                  <>

                    Заполнить пропуски с TabPFN и скачать файл
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Imputation Results Section */}
        {imputationResult && (
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-purple-200/50 overflow-hidden">
            <div className="p-6 border-b border-purple-200/50 bg-gradient-to-r from-purple-50 to-pink-50">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl">
                  <TrendingUp className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-purple-800">
                    Результаты заполнения пропусков
                  </h2>
                  <p className="text-purple-600">Статистика обработки с помощью TabPFN</p>
                </div>
              </div>
            </div>

            <div className="p-6 space-y-6">
              {/* Statistics Summary */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 ">
                <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-xl border border-green-200">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle2 className="w-5 h-5 text-green-600" />
                    <span className="font-semibold text-green-800">Обработано строк</span>
                  </div>
                  <div className="text-2xl font-bold text-green-900">{imputationResult.total_rows}</div>
                </div>

                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-xl border border-blue-200">
                  <div className="flex items-center gap-2 mb-2">
                    <Database className="w-5 h-5 text-blue-600" />
                    <span className="font-semibold text-blue-800">Столбцов обработано</span>
                  </div>
                  <div className="text-2xl font-bold text-blue-900">{Object.keys(imputationResult.missing_before).length}</div>
                </div>

                <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-xl border border-purple-200">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap className="w-5 h-5 text-purple-600" />
                    <span className="font-semibold text-purple-800">Всего заполнено</span>
                  </div>
                  <div className="text-2xl font-bold text-purple-900">
                    {Object.values(imputationResult.missing_before).reduce((a, b) => a + b, 0) -
                     Object.values(imputationResult.missing_after).reduce((a, b) => a + b, 0)}
                  </div>
                </div>
              </div>

              {/* Column-wise Results */}
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-purple-50/80">
                    <tr>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-purple-700">Столбец</th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-purple-700">Пропусков до</th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-purple-700">Пропусков после</th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-purple-700">Заполнено</th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-purple-700">Статус</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-purple-200/50">
                    {Object.keys(imputationResult.missing_before).map((col) => (
                      <tr key={col} className="hover:bg-purple-50/50 transition-colors duration-150">
                        <td className="px-6 py-4 font-medium text-gray-900">{col}</td>
                        <td className="px-6 py-4 text-amber-700 font-medium">{imputationResult.missing_before[col]}</td>
                        <td className="px-6 py-4 text-green-700 font-medium">{imputationResult.missing_after[col]}</td>
                        <td className="px-6 py-4">
                          <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                            {imputationResult.missing_before[col] - imputationResult.missing_after[col]}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          {imputationResult.processing_results[col] === "success" ? (
                            <div className="flex items-center gap-2 text-green-600">
                              <CheckCircle2 className="w-4 h-4" />
                              <span className="text-sm font-medium">Успешно</span>
                            </div>
                          ) : (
                            <div className="flex items-center gap-2 text-red-600">
                              <AlertTriangle className="w-4 h-4" />
                              <span className="text-sm font-medium">Ошибка</span>
                            </div>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Preview of Processed Data */}
              {imputationResult.preview.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">Предварительный просмотр обработанных данных</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full border border-gray-200 rounded-lg">
                      <thead className="bg-gray-50">
                        <tr>
                          {Object.keys(imputationResult.preview[0]).map((key, idx) => (
                            <th key={idx} className="px-4 py-3 text-left text-sm font-semibold text-gray-700 border-b">
                              {key}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {imputationResult.preview.slice(0, 5).map((row, idx) => (
                          <tr key={idx} className="hover:bg-gray-50 transition-colors duration-150">
                            {Object.values(row).map((val, i) => (
                              <td key={i} className="px-4 py-3 text-sm text-gray-900">
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
        )}

        {/* Outlier Detection Section */}
        {columns.length > 0 && (
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-gradient-to-r from-red-500 to-pink-500 rounded-xl">
                <Filter className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-800">Поиск выбросов</h2>
                <p className="text-gray-600">Выберите числовые столбцы для анализа</p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-h-[350px] overflow-auto">
                {columns
                  .filter((col) => col.dtype === "int64" || col.dtype === "float64")
                  .map((col) => (
                    <label key={col.column} className="relative">
                      <input
                        type="checkbox"
                        value={col.column}
                        checked={selectedColumns.includes(col.column)}
                        onChange={(e) => {
                          const val = e.target.value;
                          setSelectedColumns((prev) =>
                            prev.includes(val)
                              ? prev.filter((c) => c !== val)
                              : [...prev, val]
                          );
                        }}
                        className="sr-only"
                      />
                      <div className={`p-4 rounded-xl border-2 cursor-pointer transition-all duration-200 ${
                        selectedColumns.includes(col.column)
                          ? "border-blue-500 bg-blue-50 shadow-md"
                          : "border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm"
                      }`}>
                        <div className="flex items-center gap-3">
                          <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                            selectedColumns.includes(col.column)
                              ? "border-blue-500 bg-blue-500"
                              : "border-gray-300"
                          }`}>
                            {selectedColumns.includes(col.column) && (
                              <CheckCircle2 className="w-3 h-3 text-white" />
                            )}
                          </div>
                          <span className="font-medium text-gray-900">{col.column}</span>
                        </div>
                        <div className="mt-2 text-sm text-gray-500">
                          {col.unique} уникальных значений
                        </div>
                      </div>
                    </label>
                  ))}
              </div>

              <button
                onClick={handleDetectOutliers}
                disabled={selectedColumns.length === 0 || isAnalyzing}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 disabled:from-gray-400 disabled:to-gray-500 text-white rounded-xl font-medium transition-all duration-200 hover:shadow-lg transform hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed"
              >
                {isAnalyzing ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    Анализируем...
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-5 h-5" />
                    Найти выбросы
                  </>
                )}
              </button>

            </div>
          </div>
        )}

        {/* Data Preview Section */}
        {preview.length > 0 && (
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 overflow-hidden">
            <div className="p-6 border-b border-gray-200/50 bg-gradient-to-r from-indigo-50 to-blue-50">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-indigo-500 to-blue-500 rounded-xl">
                  <Eye className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-gray-800">Предварительный просмотр</h2>
                  <p className="text-gray-600">Первые несколько строк данных</p>
                </div>
              </div>
            </div>

            <div className="border border-gray-300 rounded-md max-h-[350px] overflow-auto">
              <table className="min-w-[1000px] w-full border-collapse">
                <thead className="bg-gray-50/80 sticky top-0 z-10">
                  <tr>
                    {Object.keys(preview[0]).map((key, idx) => (
                      <th key={idx} className="px-4 py-3 text-left text-sm font-semibold text-gray-700 border-b border-r">
                        {key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200/50">
                  {preview.map((row, idx) => (
                    <tr key={idx} className="hover:bg-gray-50/50 transition-colors duration-150">
                      {Object.values(row).map((val, i) => (
                        <td key={i} className="px-6 py-4 text-gray-900">
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

        {/* Outliers Results Section */}
        {outlierCount !== null && (
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-red-200/50 overflow-hidden">
            <div className="p-6 border-b border-red-200/50 bg-gradient-to-r from-red-50 to-pink-50">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-red-500 to-pink-500 rounded-xl">
                  <AlertTriangle className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-red-800">
                    Обнаружено выбросов: {outlierCount}
                  </h2>
                  <p className="text-red-600">Аномальные значения в выбранных столбцах</p>
                </div>
              </div>
            </div>

            {outliers && outliers.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-red-50/80">
                    <tr>
                      {Object.keys(outliers[0]).map((key, idx) => (
                        <th key={idx} className="px-6 py-4 text-left text-sm font-semibold text-red-700">
                          {key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-red-200/50">
                    {outliers.map((row, idx) => (
                      <tr key={idx} className="hover:bg-red-50/50 transition-colors duration-150">
                        {Object.values(row).map((val, i) => (
                          <td key={i} className="px-6 py-4 text-red-900 font-medium">
                            {String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            { outliers && outliers.length === 0 && outlierCount === 0 && (
                <div className="p-6 text-center text-gray-600">
                    Нет выбросов, обнаруженных в выбранных столбцах.
                </div>
            )}
          </div>
        )}

        {file && (selectedColumns.length > 0 || selectedMissingColumns.length > 0) && (
  <div className="flex justify-end mt-8">
  </div>
)}
      </div>
    </div>
  );
};

export default App;