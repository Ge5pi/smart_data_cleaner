import React, { useState, useContext } from 'react';
import { AppContext } from '../contexts/AppContext';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { Upload, FileText, AlertTriangle, BarChart3, Database, CheckCircle2, Eye, Filter, Zap, TrendingUp, Loader } from "lucide-react";

// Определим тип для результатов импутации, чтобы было проще работать
type ImputationResult = {
  preview: any[];
  missing_before: { [key: string]: number };
  missing_after: { [key: string]: number };
  processing_results: { [key: string]: string };
  total_rows: number;
};

const DataCleanerPage = () => {
    // Получаем глобальное состояние и функции из AppContext
    const {
        file, setFile, fileId, setFileId, columns, setColumns, preview, setPreview,
        isLoading, setIsLoading, error, setError, resetState
    } = useContext(AppContext)!;

    // Локальные состояния, которые используются только на этой странице
    const [selectedOutlierCols, setSelectedOutlierCols] = useState<string[]>([]);
    const [selectedMissingCols, setSelectedMissingCols] = useState<string[]>([]);
    const [isAnalyzing, setIsAnalyzing] = useState(false); // для поиска выбросов
    const [isImputing, setIsImputing] = useState(false);   // для заполнения пропусков
    const [imputationResult, setImputationResult] = useState<ImputationResult | null>(null);
    const [outliers, setOutliers] = useState<any[]>([]);
    const [outlierCount, setOutlierCount] = useState<number | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files?.[0]) {
            resetState(); // Сбрасываем глобальное состояние (fileId, etc.)
            setFile(e.target.files[0]);
            // Сбрасываем локальное состояние этой страницы
            setSelectedOutlierCols([]);
            setSelectedMissingCols([]);
            setImputationResult(null);
            setOutliers([]);
            setOutlierCount(null);
        }
    };

    const handleUpload = async () => {
        if (!file) return;
        setIsLoading(true);
        setError(null);
        const uploadFormData = new FormData();
        uploadFormData.append("file", file);
        try {
            const uploadRes = await axios.post("http://localhost:5643/upload/", uploadFormData);
            const newFileId = uploadRes.data.file_id;
            setFileId(newFileId);
            setPreview(uploadRes.data.preview);
            const analysisFormData = new FormData();
            analysisFormData.append("file_id", newFileId);
            const analysisRes = await axios.post("http://localhost:5643/analyze/", analysisFormData);
            setColumns(analysisRes.data.columns);
        } catch (err) {
            console.error(err);
            setError("Не удалось загрузить или проанализировать файл.");
        } finally {
            setIsLoading(false);
        }
    };

    const handleImpute = async () => {
        if (!fileId || selectedMissingCols.length === 0) return;
        setIsImputing(true);
        setImputationResult(null);
        const formData = new FormData();
        formData.append("file_id", fileId);
        formData.append("columns", JSON.stringify(selectedMissingCols));
        try {
            const res = await axios.post("http://localhost:5643/impute-missing/", formData);
            setImputationResult(res.data);
            alert('Пропуски успешно заполнены! Результаты показаны ниже.');
        } catch (err) {
            alert('Ошибка при заполнении пропусков.');
            console.error(err);
        } finally {
            setIsImputing(false);
        }
    };

    const handleDetectOutliers = async () => {
        if (!fileId || selectedOutlierCols.length === 0) return;
        setIsAnalyzing(true);
        setOutlierCount(null);
        setOutliers([]);
        const formData = new FormData();
        formData.append("file_id", fileId);
        formData.append("columns", JSON.stringify(selectedOutlierCols));
        try {
            const res = await axios.post("http://localhost:5643/outliers/", formData);
            setOutliers(res.data.outlier_preview);
            setOutlierCount(res.data.outlier_count);
        } catch (err) {
            console.error(err);
            setError("Не удалось определить выбросы.");
        } finally {
            setIsAnalyzing(false);
        }
    };

    const handleDownload = () => {
        if (!fileId) return;
        window.open(`http://localhost:5643/download-cleaned/${fileId}`, '_blank');
    };

    const getTypeColor = (dtype: string) => {
        if (dtype.includes("int") || dtype.includes("float")) return "bg-blue-100 text-blue-800";
        if (dtype.includes("object")) return "bg-green-100 text-green-800";
        return "bg-gray-100 text-gray-800";
    };

    const getTypeIcon = (dtype: string) => {
        if (dtype.includes("int") || dtype.includes("float")) return <BarChart3 className="w-4 h-4" />;
        if (dtype.includes("object")) return <FileText className="w-4 h-4" />;
        return <Database className="w-4 h-4" />;
    };

    return (
        <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
            {/* Upload Section */}
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-8">
                <div className="flex items-center gap-4 mb-6">
                    <div className="p-3 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-xl"><Upload className="w-6 h-6 text-white" /></div>
                    <div><h2 className="text-xl font-semibold text-gray-800">1. Загрузка данных</h2><p className="text-gray-600">Выберите CSV файл для анализа и очистки</p></div>
                </div>
                <div className="flex items-center gap-4">
                    <label htmlFor="file-upload" className="flex items-center gap-2 px-6 py-3 bg-white border-2 rounded-xl cursor-pointer hover:border-blue-500 transition-colors">
                        <FileText className="text-gray-600" /> <span className="text-gray-700 font-medium">{file ? file.name : "Выбрать файл"}</span>
                        <input id="file-upload" type="file" accept=".csv" onChange={handleFileChange} className="hidden" />
                    </label>
                    <button onClick={handleUpload} disabled={!file || isLoading} className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-xl disabled:bg-gray-400 transition-colors">
                        {isLoading ? <><Loader className="w-5 h-5 animate-spin" /><span>Загрузка...</span></> : <><Upload className="w-5 h-5" /><span>Загрузить</span></>}
                    </button>
                </div>
                {error && <p className="mt-4 text-red-600 bg-red-100 p-3 rounded-lg">{error}</p>}
                {fileId && (
                    <div className="mt-6 p-4 bg-indigo-50 rounded-lg border border-indigo-200">
                        <h3 className="font-semibold text-indigo-800">Файл успешно загружен!</h3>
                        <p className="text-indigo-700 mt-1">Теперь вы можете использовать инструменты ниже или перейти к диалогу с AI-агентом для продвинутого анализа.</p>
                        <Link to="/chat" className="mt-3 inline-block px-5 py-2 bg-indigo-600 text-white font-semibold rounded-lg hover:bg-indigo-700 transition-colors shadow-sm">
                            Перейти к AI Агенту →
                        </Link>
                    </div>
                )}
            </div>

            {/* Column Analysis Section */}
            {columns.length > 0 && (
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 overflow-hidden">
                <div className="p-6 border-b border-gray-200/50 bg-gradient-to-r from-green-50 to-emerald-50">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl"><BarChart3 className="w-6 h-6 text-white" /></div>
                        <div><h2 className="text-xl font-semibold text-gray-800">Анализ столбцов</h2><p className="text-gray-600">Обзор структуры данных</p></div>
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
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200/50">
                            {columns.map((col) => (
                            <tr key={col.column} className="hover:bg-gray-50/50">
                                <td className="px-6 py-4 font-medium text-gray-900">{col.column}</td>
                                <td className="px-6 py-4"><div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium ${getTypeColor(col.dtype)}`}>{getTypeIcon(col.dtype)} {col.dtype}</div></td>
                                <td className="px-6 py-4"><div className={`flex items-center gap-2 ${col.nulls > 0 ? "text-amber-700" : "text-green-700"}`}>{col.nulls > 0 ? <AlertTriangle className="w-4 h-4 text-amber-500" /> : <CheckCircle2 className="w-4 h-4 text-green-500" />} {col.nulls}</div></td>
                                <td className="px-6 py-4 text-gray-900">{col.unique}</td>
                            </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
            )}

            {/* Missing Values Imputation Section */}
            {columns.length > 0 && columns.some(col => col.nulls > 0) && (
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-6">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl"><Zap className="w-6 h-6 text-white" /></div>
                    <div><h2 className="text-xl font-semibold text-gray-800">Заполнение пропусков с TabPFN</h2><p className="text-gray-600">Выберите столбцы для интеллектуального заполнения</p></div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {columns.filter((col) => col.nulls > 0).map((col) => (
                    <label key={col.column} className="relative">
                        <input type="checkbox" value={col.column} checked={selectedMissingCols.includes(col.column)}
                            onChange={(e) => {
                                const val = e.target.value;
                                setSelectedMissingCols((prev) => prev.includes(val) ? prev.filter((c) => c !== val) : [...prev, val]);
                            }} className="sr-only" />
                        <div className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${selectedMissingCols.includes(col.column) ? "border-purple-500 bg-purple-50" : "border-gray-200 bg-white hover:border-gray-300"}`}>
                            <div className="font-medium text-gray-900">{col.column}</div>
                            <div className="text-sm text-amber-600">{col.nulls} пропусков</div>
                        </div>
                    </label>
                    ))}
                </div>
                <button onClick={handleImpute} disabled={isImputing || selectedMissingCols.length === 0} className="mt-6 flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-xl disabled:bg-gray-400 transition-colors">
                    {isImputing ? <><Loader className="w-5 h-5 animate-spin" /><span>Заполнение...</span></> : "Заполнить пропуски"}
                </button>
            </div>
            )}

            {/* Imputation Results Section */}
            {imputationResult && (
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-purple-200/50 overflow-hidden">
                <div className="p-6 border-b border-purple-200/50 bg-gradient-to-r from-purple-50 to-pink-50">
                    <div className="flex items-center gap-3"><div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl"><TrendingUp className="w-6 h-6 text-white" /></div>
                    <div><h2 className="text-xl font-semibold text-purple-800">Результаты заполнения</h2><p className="text-purple-600">Статистика обработки</p></div></div>
                </div>
                <div className="p-6">
                    {/* ... (таблица с результатами импутации, как в вашем файле) ... */}
                </div>
            </div>
            )}

            {/* Outlier Detection Section */}
            {columns.length > 0 && (
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-6">
                 <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-gradient-to-r from-red-500 to-pink-500 rounded-xl"><Filter className="w-6 h-6 text-white" /></div>
                    <div><h2 className="text-xl font-semibold text-gray-800">Поиск выбросов</h2><p className="text-gray-600">Выберите числовые столбцы для анализа</p></div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {columns.filter((col) => col.dtype === "int64" || col.dtype === "float64").map((col) => (
                    <label key={col.column} className="relative">
                        <input type="checkbox" value={col.column} checked={selectedOutlierCols.includes(col.column)}
                            onChange={(e) => {
                                const val = e.target.value;
                                setSelectedOutlierCols((prev) => prev.includes(val) ? prev.filter((c) => c !== val) : [...prev, val]);
                            }} className="sr-only" />
                        <div className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${selectedOutlierCols.includes(col.column) ? "border-red-500 bg-red-50" : "border-gray-200 bg-white hover:border-gray-300"}`}>
                            <div className="font-medium text-gray-900">{col.column}</div>
                            <div className="text-sm text-gray-500">{col.unique} уникальных</div>
                        </div>
                    </label>
                ))}
                </div>
                <button onClick={handleDetectOutliers} disabled={isAnalyzing || selectedOutlierCols.length === 0} className="mt-6 flex items-center gap-2 px-6 py-3 bg-red-600 text-white rounded-xl disabled:bg-gray-400 transition-colors">
                    {isAnalyzing ? <><Loader className="w-5 h-5 animate-spin" /><span>Анализ...</span></> : "Найти выбросы"}
                </button>
            </div>
            )}

            {/* Outliers Results Section */}
            {outlierCount !== null && (
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-red-200/50 overflow-hidden">
                <div className="p-6 border-b border-red-200/50 bg-gradient-to-r from-red-50 to-pink-50">
                    <div className="flex items-center gap-3"><div className="p-2 bg-gradient-to-r from-red-500 to-pink-500 rounded-xl"><AlertTriangle className="w-6 h-6 text-white" /></div>
                    <div><h2 className="text-xl font-semibold text-red-800">Обнаружено выбросов: {outlierCount}</h2><p className="text-red-600">Аномальные значения в данных</p></div></div>
                </div>
                {outliers.length > 0 ? (
                    <div className="overflow-x-auto"><table className="w-full">
                        <thead className="bg-red-50/80"><tr>{Object.keys(outliers[0]).map((key) => <th key={key} className="px-6 py-4 text-left text-sm font-semibold text-red-700">{key}</th>)}</tr></thead>
                        <tbody className="divide-y divide-red-200/50">{outliers.map((row, idx) => (<tr key={idx}>{Object.values(row).map((val: any, i) => <td key={i} className="px-6 py-4 text-red-900">{String(val)}</td>)}</tr>))}</tbody>
                    </table></div>
                ) : <div className="p-6 text-center text-gray-600">Выбросы не найдены в выбранных столбцах.</div>}
            </div>
            )}

            {/* Data Preview Section */}
            {preview.length > 0 && (
                <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 overflow-hidden">
                    <div className="p-6 border-b border-gray-200/50"><h2 className="text-xl font-semibold text-gray-800">Предпросмотр данных</h2></div>
                    <div className="overflow-x-auto"><table className="w-full">
                        <thead className="bg-gray-50/80"><tr>{Object.keys(preview[0]).map((key) => <th key={key} className="px-6 py-4 text-left text-sm font-semibold text-gray-700">{key}</th>)}</tr></thead>
                        <tbody className="divide-y divide-gray-200/50">{preview.map((row, idx) => (<tr key={idx} className="hover:bg-gray-50/50">{Object.values(row).map((val: any, i) => <td key={i} className="px-6 py-4 text-gray-800">{String(val)}</td>)}</tr>))}</tbody>
                    </table></div>
                </div>
            )}

            {/* Download Section */}
            {fileId && (
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-6">
                <h2 className="text-xl font-semibold mb-2">Скачать обработанный файл</h2>
                <p className="text-sm text-gray-600 mb-4">Будет скачана последняя сохраненная на сервере версия файла. Убедитесь, что вы применили все необходимые операции по очистке.</p>
                <button onClick={handleDownload} className="px-6 py-3 bg-green-600 text-white rounded-xl font-medium">
                    Скачать файл
                </button>
            </div>
            )}
        </div>
    );
};

export default DataCleanerPage;