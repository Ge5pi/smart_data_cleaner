import React, { useState } from "react";
import { Upload, FileText, AlertTriangle, BarChart3, Database, CheckCircle2, Eye, Filter } from "lucide-react"; // Importing icons from lucide-react
import axios from "axios";
type ColumnAnalysis = {
  column: string;
  dtype: string;
  nulls: number;
  unique: number;
  sample_values: string[];
};

const App = () => {
  // State variables for the application
  const [outliers, setOutliers] = useState<any[]>([]); // Stores detected outlier rows
  const [outlierCount, setOutlierCount] = useState<number | null>(null); // Stores the total count of outliers
  const [preview, setPreview] = useState<any[]>([]); // Stores a preview of the uploaded data
  const [columns, setColumns] = useState<ColumnAnalysis[]>([]); // Stores the analysis of each column
  const [file, setFile] = useState<File | null>(null); // Stores the selected file object
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]); // Stores selected columns for outlier detection
  const [isLoading, setIsLoading] = useState(false); // Indicates if a file upload is in progress
  const [isAnalyzing, setIsAnalyzing] = useState(false); // Indicates if outlier detection is in progress

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
    }
  };

  // Handles the file upload and initial data analysis
  const handleUpload = async () => {
    if (!file) return; // Do nothing if no file is selected
    setIsLoading(true); // Set loading state to true

    const formData = new FormData(); // Create a new FormData object to send the file
    formData.append("file", file); // Append the file to the form data

    try {
      // Send the file for preview
      const previewRes = await axios.post("http://localhost:5643/upload/", formData);
      setPreview(previewRes.data.preview); // Update preview state with response data

      // Send the file for column analysis
      const analysisRes = await axios.post("http://localhost:5643/analyze/", formData);
      setColumns(analysisRes.data.columns); // Update columns state with response data
    } catch (error) {
      console.error("Ошибка при загрузке файла:", error); // Log any errors during upload
    } finally {
      setIsLoading(false); // Set loading state to false after upload completes or fails
    }
  };

  // Handles the outlier detection process
  const handleDetectOutliers = async () => {
    if (!file || selectedColumns.length === 0) return; // Do nothing if no file or columns are selected
    setIsAnalyzing(true); // Set analyzing state to true

    const formData = new FormData(); // Create new FormData for outlier detection
    formData.append("file", file); // Append the file
    // Append selected columns as a JSON string
    formData.append("columns", JSON.stringify(selectedColumns));

    try {
      // Send data for outlier detection
      const res = await axios.post("http://localhost:5643/outliers/", formData);
      setOutliers(res.data.outlier_preview); // Update outliers state
      setOutlierCount(res.data.outlier_count); // Update outlier count state
    } catch (error) {
      console.error("Ошибка при определении выбросов:", error); // Log any errors
    } finally {
      setIsAnalyzing(false); // Set analyzing state to false
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
        return <BarChart3 className="w-4 h-4" />; // Bar chart icon for numerical types
      case "object":
        return <FileText className="w-4 h-4" />; // File text icon for object types (strings)
      default:
        return <Database className="w-4 h-4" />; // Database icon for other types
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

            <div className="overflow-x-auto">
              <table className="w-full">
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
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {columns
                  .filter((col) => col.dtype === "int64" || col.dtype === "float64") // Only show numerical columns
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
                              ? prev.filter((c) => c !== val) // Remove if already selected
                              : [...prev, val] // Add if not selected
                          );
                        }}
                        className="sr-only" // Hide default checkbox
                      />
                      <div className={`p-4 rounded-xl border-2 cursor-pointer transition-all duration-200 ${
                        selectedColumns.includes(col.column)
                          ? "border-blue-500 bg-blue-50 shadow-md" // Style for selected checkbox
                          : "border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm" // Style for unselected checkbox
                      }`}>
                        <div className="flex items-center gap-3">
                          <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                            selectedColumns.includes(col.column)
                              ? "border-blue-500 bg-blue-500" // Background for checked state
                              : "border-gray-300" // Border for unchecked state
                          }`}>
                            {selectedColumns.includes(col.column) && (
                              <CheckCircle2 className="w-3 h-3 text-white" /> // Check icon when selected
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

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50/80">
                  <tr>
                    {/* Render table headers from the keys of the first row in preview data */}
                    {Object.keys(preview[0]).map((key, idx) => (
                      <th key={idx} className="px-6 py-4 text-left text-sm font-semibold text-gray-700">
                        {key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200/50">
                  {/* Render table rows from preview data */}
                  {preview.map((row, idx) => (
                    <tr key={idx} className="hover:bg-gray-50/50 transition-colors duration-150">
                      {/* Render table cells from the values of each row */}
                      {Object.values(row).map((val, i) => (
                        <td key={i} className="px-6 py-4 text-gray-900">
                          {/* Convert value to string for display */}
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
        {outlierCount !== null && ( // Conditionally render if outlierCount is available
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

            {outliers.length > 0 && ( // Conditionally render outlier table if there are outliers
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-red-50/80">
                    <tr>
                      {/* Render table headers from the keys of the first outlier row */}
                      {Object.keys(outliers[0]).map((key, idx) => (
                        <th key={idx} className="px-6 py-4 text-left text-sm font-semibold text-red-700">
                          {key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-red-200/50">
                    {/* Render outlier rows */}
                    {outliers.map((row, idx) => (
                      <tr key={idx} className="hover:bg-red-50/50 transition-colors duration-150">
                        {/* Render cells for each outlier value */}
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
            {outliers.length === 0 && outlierCount === 0 && (
                <div className="p-6 text-center text-gray-600">
                    Нет выбросов, обнаруженных в выбранных столбцах.
                </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
