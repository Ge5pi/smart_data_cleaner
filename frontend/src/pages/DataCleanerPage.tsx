import { useState, useContext, useEffect } from 'react';
import { AppContext } from '../contexts/AppContext';
import { Link } from 'react-router-dom';
import { Upload, FileText, AlertTriangle, BarChart3, Database, CheckCircle2, Filter, Zap, TrendingUp, Loader, History, FileClock } from "lucide-react";
import { format } from 'date-fns';
import api from '../api';

type ImputationResult = {
  preview: any[];
  missing_before: { [key: string]: number };
  missing_after: { [key: string]: number };
  processing_results: { [key: string]: string };
  total_rows: number;
};

const DataCleanerPage = () => {
    const {
        file, setFile, fileId, setFileId, columns, setColumns, preview, setPreview,
        isLoading, setIsLoading, error, setError, resetState,
        token,
        userFiles, setUserFiles
    } = useContext(AppContext)!;

    // Локальные состояния, которые используются только на этой странице
    const [selectedOutlierCols, setSelectedOutlierCols] = useState<string[]>([]);
    const [selectedMissingCols, setSelectedMissingCols] = useState<string[]>([]);
    const [isAnalyzing, setIsAnalyzing] = useState(false); // для поиска выбросов
    const [isImputing, setIsImputing] = useState(false);   // для заполнения пропусков
    const [imputationResult, setImputationResult] = useState<ImputationResult | null>(null);
    const [outliers, setOutliers] = useState<any[]>([]);
    const [outlierCount, setOutlierCount] = useState<number | null>(null);
    const [selectedEncodingCols, setSelectedEncodingCols] = useState<string[]>([]);
    const [isEncoding, setIsEncoding] = useState(false);
    const [currentPage, setCurrentPage] = useState(1);
    const rowsPerPage = 50;
    const [totalRows, setTotalRows] = useState(0);


    useEffect(() => {
        if (fileId) { // Токен уже проверяется в api.ts, можно убрать
            setIsLoading(true);
            // Убираем лишний headers, interceptor в api.ts сделает всё сам
            api.get(`/preview/${fileId}`, {
                params: { page: currentPage, page_size: rowsPerPage }
            })
            .then(res => {
                setPreview(res.data.preview);
                setTotalRows(res.data.total_rows);
            })
            .catch(_ => {
                setError("Не удалось загрузить предпросмотр данных.");
            })
            .finally(() => setIsLoading(false));
        }
    }, [fileId, currentPage]);


    useEffect(() => {
        const fetchUserFiles = async () => {
            if (token) {
                try {
                    // Убираем headers
                    const res = await api.get('/files/me');
                    setUserFiles(res.data);
                } catch (err) {
                    console.error("Не удалось загрузить список файлов:", err);
                }
            }
        };
        fetchUserFiles();
    }, [token, setUserFiles]);

    // --- 2. ФУНКЦИЯ ВЫБОРА И АНАЛИЗА СУЩЕСТВУЮЩЕГО ФАЙЛА ---
    const handleSelectFile = async (selectedFileId: string, selectedFileName: string) => {
        resetState(); // Сбрасываем все предыдущие состояния
        setIsLoading(true);
        setError(null);

        // Создаем настоящий, но пустой объект File для консистентности типа в состоянии
        const placeholderFile = new File([""], selectedFileName, { type: "text/csv"});
        setFile(placeholderFile);

        const formData = new FormData();
        formData.append("file_id", selectedFileId);

        try {
            // Убираем headers
            const res = await api.post('/analyze-existing/', formData);
            setFileId(selectedFileId);
            setColumns(res.data.columns);
            setPreview(res.data.preview);
            setTotalRows(res.data.total_rows || res.data.preview.length);
            setCurrentPage(1);
        } catch (err: any) {
            // --- ИЗМЕНЕНИЕ ---
            const message = err.response?.data?.detail || "Не удалось проанализировать выбранный файл.";
            setError(message);
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    const handleEncode = async () => {
        if (!fileId || selectedEncodingCols.length === 0 || !token) return;

        setIsEncoding(true);
        setError(null);
        const formData = new FormData();
        formData.append("file_id", fileId);
        formData.append("columns", JSON.stringify(selectedEncodingCols));

        try {
            const res = await api.post(`/encode-categorical/`, formData);

            // Обновляем состояние страницы новыми данными с сервера
            setColumns(res.data.columns);
            setPreview(res.data.preview);

            // Сбрасываем выбор и результаты других операций
            setSelectedEncodingCols([]);
            setImputationResult(null);
            setOutlierCount(null);

            alert(res.data.message);
        } catch (err: any) {
            setError(err.response?.data?.detail || "Ошибка при кодировании столбцов.");
        } finally {
            setIsEncoding(false);
        }
    };

    // --- 3. ФУНКЦИЯ ВЫБОРА НОВОГО ФАЙЛА С ДИСКА ---
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

    // --- 4. ФУНКЦИЯ ЗАГРУЗКИ НОВОГО ФАЙЛА НА СЕРВЕР ---
    const handleUpload = async () => {
        if (!file) return;
        setIsLoading(true);
        setError(null);

        // Используем токен из контекста, а не из localStorage напрямую
        if (!token) {
            setError("Ошибка аутентификации: токен не найден. Пожалуйста, войдите в систему заново.");
            setIsLoading(false);
            return;
        }

        const uploadFormData = new FormData();
        uploadFormData.append("file", file);

        try {
            const uploadRes = await api.post("/upload/", uploadFormData);

            // Обновляем список файлов в UI после успешной загрузки
            const newFile = {
                file_uid: uploadRes.data.file_id,
                file_name: file.name,
                datetime_created: new Date().toISOString()
            };
            setUserFiles(prev => [newFile, ...prev]);

            // Сразу анализируем только что загруженный файл, вызвав другую нашу функцию
            await handleSelectFile(uploadRes.data.file_id, file.name);

        }
        catch (err: any) {
            // --- ИЗМЕНЕНИЕ ---
            const message = err.response?.data?.detail || "Не удалось загрузить файл.";
            setError(message);
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    // --- 5. ОСТАЛЬНЫЕ ОБРАБОТЧИКИ ---
    const handleImpute = async () => {
        if (!fileId || selectedMissingCols.length === 0) return;
        setIsImputing(true);
        setImputationResult(null);
        const formData = new FormData();
        formData.append("file_id", fileId);
        formData.append("columns", JSON.stringify(selectedMissingCols));
        try {
            const res = await api.post(`/impute-missing/`, formData);
            setImputationResult(res.data);
            alert('Пропуски успешно заполнены!');
        } catch (err: any) {
            // --- ИЗМЕНЕНИЕ ---
            const message = err.response?.data?.detail || "Ошибка при заполнении пропусков.";
            alert(message); // Здесь можно оставить alert или использовать setError
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
            const res = await api.post(`/outliers/`, formData);
            setOutliers(res.data.outlier_preview);
            setOutlierCount(res.data.outlier_count);
        } catch (err: any) {
            // --- ИЗМЕНЕНИЕ ---
            const message = err.response?.data?.detail || "Не удалось определить выбросы.";
            setError(message);
            console.error(err);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const handleDownload = async () => {
        if (!fileId) return;

        try {
            const response = await api.get(`/download-cleaned/${fileId}`, {
                responseType: 'blob',
            });

            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;

            const currentFileName = file?.name || 'cleaned_data.csv';
            link.setAttribute('download', currentFileName);

            document.body.appendChild(link);
            link.click();

            link.parentNode?.removeChild(link);
            window.URL.revokeObjectURL(url);

        } catch (error) {
            console.error('Ошибка при скачивании файла:', error);
            setError('Не удалось скачать файл.');
        }
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
        <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                <aside className="lg:col-span-1">
                {/* --- Сайдбар (без изменений) --- */}
                <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-6 sticky top-28">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-2 bg-gradient-to-r from-gray-500 to-gray-700 rounded-xl"><History className="w-6 h-6 text-white" /></div>
                        <h2 className="text-xl font-semibold text-gray-800">Ваши файлы</h2>
                    </div>
                    {/* --- ДОБАВЛЕНЫ КЛАССЫ ДЛЯ СКРОЛЛА --- */}
                    <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-gray-200/50">
                        {userFiles.length > 0 ? userFiles.map(f => (
                            <button
                                key={f.file_uid}
                                onClick={() => handleSelectFile(f.file_uid, f.file_name)}
                                className={`w-full text-left p-3 rounded-lg transition-colors flex items-start gap-3 ${fileId === f.file_uid ? 'bg-blue-100 border-blue-300' : 'bg-gray-50 hover:bg-gray-100'}`}
                            >
                                <FileClock className="w-5 h-5 text-gray-500 mt-0.5 flex-shrink-0" />
                                <div>
                                    <p className="font-medium text-gray-800 break-all">{f.file_name}</p>
                                    <p className="text-xs text-gray-500">{format(new Date(f.datetime_created), 'dd.MM.yyyy HH:mm')}</p>
                                </div>
                            </button>
                        )) : (
                            <p className="text-sm text-gray-500 text-center py-4">Вы еще не загружали файлы.</p>
                        )}
                    </div>
                </div>
            </aside>

                {/* --- ОСНОВНОЙ КОНТЕНТ --- */}
                <main className="lg:col-span-3 space-y-8">
                    {/* Upload Section */}
                    <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-8">
                        <div className="flex items-center gap-4 mb-6">
                            <div className="p-3 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-xl"><Upload className="w-6 h-6 text-white" /></div>
                            <div><h2 className="text-xl font-semibold text-gray-800">1. Загрузка и выбор данных</h2><p className="text-gray-600">Загрузите новый CSV или Excel файл или выберите существующий слева</p></div>
                        </div>
                        <div className="flex items-center gap-4">
                            <label htmlFor="file-upload" className="flex items-center gap-2 px-6 py-3 bg-white border-2 rounded-xl cursor-pointer hover:border-blue-500 transition-colors">
                                <FileText className="text-gray-600" /> <span className="text-gray-700 font-medium">{file ? file.name : "Выбрать файл"}</span>
                                <input id="file-upload" type="file" accept=".csv, .xlsx, .xls" onChange={handleFileChange} className="hidden" />
                            </label>
                            <button onClick={handleUpload} disabled={!file || isLoading} className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-xl disabled:bg-gray-400 transition-colors">
                                {isLoading ? <><Loader className="w-5 h-5 animate-spin" /><span>Загрузка...</span></> : <><Upload className="w-5 h-5" /><span>Загрузить</span></>}
                            </button>
                        </div>
                        {error && <p className="mt-4 text-red-600 bg-red-100 p-3 rounded-lg">{error}</p>}
                        {fileId && (
                            <div className="mt-6 p-4 bg-indigo-50 rounded-lg border border-indigo-200">
                                <h3 className="font-semibold text-indigo-800">Файл "{file?.name}" готов к анализу!</h3>
                                <p className="text-indigo-700 mt-1">Теперь вы можете использовать инструменты ниже или перейти к диалогу с AI-агентом.</p>
                                <Link to="/chat" className="mt-3 inline-block px-5 py-2 bg-indigo-600 text-white font-semibold rounded-lg hover:bg-indigo-700 transition-colors shadow-sm">
                                    Перейти к AI Агенту →
                                </Link>
                            </div>
                        )}
                    </div>

                    {/* Сообщение-приветствие, если файл не выбран */}
                    {!fileId && (
                        <div className="text-center py-16 px-6 bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50">
                            <Database className="w-16 h-16 mx-auto text-gray-300" />
                            <h2 className="mt-4 text-2xl font-semibold text-gray-800">Начните работу</h2>
                            <p className="mt-2 text-gray-500">Загрузите новый файл или выберите один из ранее загруженных в панели слева.</p>
                        </div>
                    )}

                    {/* Все остальные секции анализа, которые показываются только если есть fileId */}
                    {fileId && (
                        <>
                            {/* Column Analysis Section */}
                            {columns.length > 0 && (
                            <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 overflow-hidden">
                                <div className="p-6 border-b border-gray-200/50 bg-gradient-to-r from-green-50 to-emerald-50">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl"><BarChart3 className="w-6 h-6 text-white" /></div>
                                        <div><h2 className="text-xl font-semibold text-gray-800">Анализ столбцов</h2><p className="text-gray-600">Обзор структуры данных</p></div>
                                    </div>
                                </div>
                                <div className="overflow-x-auto max-h-[400px] overflow-y-auto scrollbar-thin scrollbar-thumb-green-400 scrollbar-track-green-100">
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
                                <div className="max-h-[250px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-purple-400 scrollbar-track-purple-100">
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
                                </div>
                                <button onClick={handleImpute} disabled={isImputing || selectedMissingCols.length === 0} className="mt-6 flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-xl disabled:bg-gray-400 transition-colors">
                                    {isImputing ? <><Loader className="w-5 h-5 animate-spin" /><span>Заполнение...</span></> : "Заполнить пропуски"}
                                </button>
                            </div>
                            )}
                            {imputationResult && (
                                        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-purple-200/50 overflow-hidden">
                                            <div className="p-6 border-b border-purple-200/50 bg-gradient-to-r from-purple-50 to-pink-50">
                                                <div className="flex items-center gap-3">
                                                    <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl">
                                                        <TrendingUp className="w-6 h-6 text-white" />
                                                    </div>
                                                    <div>
                                                        <h2 className="text-xl font-semibold text-purple-800">Результаты заполнения</h2>
                                                        <p className="text-purple-600">Статистика обработки для выбранных столбцов</p>
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="overflow-x-auto">
                                                <table className="w-full text-sm">
                                                    <thead className="bg-gray-50/80">
                                                        <tr>
                                                            <th className="px-6 py-4 text-left font-semibold text-gray-700">Столбец</th>
                                                            <th className="px-6 py-4 text-left font-semibold text-gray-700">Пропусков до</th>
                                                            <th className="px-6 py-4 text-left font-semibold text-gray-700">Пропусков после</th>
                                                            <th className="px-6 py-4 text-left font-semibold text-gray-700">Статус</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody className="divide-y divide-gray-200/50">
                                                        {Object.keys(imputationResult?.processing_results || {}).map((colName) => (
                                                            <tr key={colName} className="hover:bg-gray-50/50">
                                                                <td className="px-6 py-4 font-medium text-gray-900">{colName}</td>
                                                                <td className="px-6 py-4 text-gray-800">{imputationResult.missing_before[colName]}</td>
                                                                <td className={`px-6 py-4 font-semibold ${imputationResult.missing_after[colName] > 0 ? 'text-amber-600' : 'text-green-600'}`}>
                                                                    {imputationResult.missing_after[colName]}
                                                                </td>
                                                                <td className="px-6 py-4 text-gray-800">{imputationResult.processing_results[colName]}</td>
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                        )}

                            {columns.length > 0 && (
                            <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-6">
                                 <div className="flex items-center gap-3 mb-6">
                                    <div className="p-2 bg-gradient-to-r from-red-500 to-pink-500 rounded-xl"><Filter className="w-6 h-6 text-white" /></div>
                                    <div><h2 className="text-xl font-semibold text-gray-800">Поиск выбросов</h2><p className="text-gray-600">Выберите числовые столбцы для анализа</p></div>
                                </div>
                                <div className="max-h-[250px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-red-400 scrollbar-track-red-100">
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
                                    <div className="flex items-center gap-3">
                                    <div className="p-2 bg-gradient-to-r from-red-500 to-pink-500 rounded-xl"><AlertTriangle className="w-6 h-6 text-white" /></div>
                                    <div>
                                    <h2 className="text-xl font-semibold text-red-800">Обнаружено выбросов: {outlierCount}</h2>
                                    <p className="text-red-600">Аномальные значения в данных</p></div></div>
                                </div>
                                {outliers.length > 0 ? (
                                    <div className="overflow-x-auto max-h-[300px] overflow-y-auto scrollbar-thin scrollbar-thumb-red-400 scrollbar-track-red-100">
                                    <table className="w-full">
                                        <thead className="bg-red-50/80"><tr>{Object.keys(outliers[0]).map((key) => <th key={key} className="px-6 py-4 text-left text-sm font-semibold text-red-700">{key}</th>)}</tr></thead>
                                        <tbody className="divide-y divide-red-200/50">{outliers.map((row, idx) => (<tr key={idx}>{Object.values(row).map((val: any, i) => <td key={i} className="px-6 py-4 text-red-900">{String(val)}</td>)}</tr>))}</tbody>
                                    </table></div>
                                ) : <div className="p-6 text-center text-gray-600">Выбросы не найдены в выбранных столбцах.</div>}
                            </div>
                            )}

                            {columns.length > 0 && columns.some(col => col.dtype === 'object') && (
                                <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-6">
                                    <div className="flex items-center gap-3 mb-6">
                                        <div className="p-2 bg-gradient-to-r from-cyan-500 to-sky-500 rounded-xl"><Zap className="w-6 h-6 text-white" /></div>
                                        <div><h2 className="text-xl font-semibold text-gray-800">Кодирование категорий</h2><p className="text-gray-600">Преобразуйте текстовые столбцы в числовые (One-Hot)</p></div>
                                    </div>
                                    <div className="max-h-[250px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-cyan-400 scrollbar-track-cyan-100">
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                            {columns.filter((col) => col.dtype === 'object').map((col) => (
                                            <label key={col.column} className="relative">
                                                <input type="checkbox" value={col.column} checked={selectedEncodingCols.includes(col.column)}
                                                    onChange={(e) => {
                                                        const val = e.target.value;
                                                        setSelectedEncodingCols((prev) => prev.includes(val) ? prev.filter((c) => c !== val) : [...prev, val]);
                                                    }} className="sr-only" />
                                                <div className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${selectedEncodingCols.includes(col.column) ? "border-cyan-500 bg-cyan-50" : "border-gray-200 bg-white hover:border-gray-300"}`}>
                                                    <div className="font-medium text-gray-900">{col.column}</div>
                                                    <div className="text-sm text-gray-500">{col.unique} уникальных</div>
                                                </div>
                                            </label>
                                            ))}
                                        </div>
                                    </div>
                                    <button onClick={handleEncode} disabled={isEncoding || selectedEncodingCols.length === 0} className="mt-6 flex items-center gap-2 px-6 py-3 bg-cyan-600 text-white rounded-xl disabled:bg-gray-400 transition-colors">
                                        {isEncoding ? <><Loader className="w-5 h-5 animate-spin" /><span>Кодирование...</span></> : "Закодировать"}
                                    </button>
                                </div>
                                )}

                            {/* Data Preview Section */}
                            {preview.length > 0 && (
                                <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 overflow-hidden">
                                    <div className="p-6 border-b border-gray-200/50">
                                    <h2 className="text-xl font-semibold text-gray-800">Предпросмотр данных</h2></div>
                                    <div className="overflow-x-auto max-h-[500px] overflow-y-auto scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-gray-200/50">
                                    <table className="w-full">
                                        <thead className="bg-gray-50/80">
                                        <tr>{Object.keys(preview[0]).map((key) => <th key={key} className="px-6 py-4 text-left text-sm font-semibold text-gray-700">{key}</th>)}</tr></thead>
                                        <tbody className="divide-y divide-gray-200/50">{preview.map((row, idx) => (<tr key={idx} className="hover:bg-gray-50/50">{Object.values(row).map((val: any, i) => <td key={i} className="px-6 py-4 text-gray-800">{String(val)}</td>)}</tr>))}</tbody>
                                    </table></div>

                                    {totalRows > 0 && (
                                        <div className="flex items-center justify-between p-4 border-t border-gray-200/50">
                                            <span className="text-sm text-gray-700">
                                                Всего строк: <span className="font-semibold">{totalRows}</span>
                                            </span>
                                            <div className="flex items-center gap-2">
                                                <button
                                                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                                    disabled={currentPage === 1 || isLoading}
                                                    className="px-3 py-1 border rounded-md disabled:opacity-50"
                                                >
                                                    Назад
                                                </button>
                                                <span className="text-sm text-gray-700">
                                                    Страница {currentPage} из {Math.ceil(totalRows / rowsPerPage)}
                                                </span>
                                                <button
                                                    onClick={() => setCurrentPage(p => Math.min(Math.ceil(totalRows / rowsPerPage), p + 1))}
                                                    disabled={currentPage === Math.ceil(totalRows / rowsPerPage) || isLoading}
                                                    className="px-3 py-1 border rounded-md disabled:opacity-50"
                                                >
                                                    Вперед
                                                </button>
                                            </div>
                                        </div>
                                    )}
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
                        </>
                    )}
                </main>
            </div>
        </div>
    );
};

export default DataCleanerPage;