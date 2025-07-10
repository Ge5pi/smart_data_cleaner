import { useState, useContext, useEffect } from 'react';
import { AppContext } from '../contexts/AppContext';
import { Chart, Bar, Scatter, Pie, Line, Bubble } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  TimeScale
} from 'chart.js';
import { BarChart3, TrendingUp, Loader } from "lucide-react";
import api from '../api';

// --- НАЧАЛО ИСПРАВЛЕНИЯ ---
// 1. Импортируем из НОВОЙ, правильной библиотеки
import { BoxPlotController, BoxAndWiskers } from '@sgratzl/chartjs-chart-boxplot';

// 2. Регистрируем все компоненты, включая Box Plot из новой библиотеки
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  TimeScale,
  BoxPlotController,
  BoxAndWiskers
);

const ChartsPage = () => {
  const { fileId, token } = useContext(AppContext)!;

  const [columns, setColumns] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [chartType, setChartType] = useState('histogram');
  const [selectedChartCol1, setSelectedChartCol1] = useState('');
  const [selectedChartCol2, setSelectedChartCol2] = useState('');
  const [selectedChartCol3, setSelectedChartCol3] = useState('');
  const [chartData, setChartData] = useState<any>(null);
  const [isChartLoading, setIsChartLoading] = useState(false);
  const [isPageLoading, setIsPageLoading] = useState(true);

  useEffect(() => {
    if (fileId && token) {
      setIsPageLoading(true);
      const formData = new FormData();
      formData.append("file_id", fileId);

      api.post("/analyze-existing/", formData)
        .then(res => setColumns(res.data.columns))
        .catch(err => {
          console.error("Ошибка при загрузке данных", err);
          setError("Не удалось загрузить данные о столбцах.");
        })
        .finally(() => setIsPageLoading(false));
    } else {
      setIsPageLoading(false);
    }
  }, [fileId, token]);

  useEffect(() => {
    setSelectedChartCol1('');
    setSelectedChartCol2('');
    setSelectedChartCol3('');
    setChartData(null);
  }, [chartType, fileId]);

  const handleGenerateChart = async () => {
    const col1Missing = !selectedChartCol1;
    const col2Missing = ['scatter', 'line', 'area', 'bubble'].includes(chartType) && !selectedChartCol2;
    const col3Missing = chartType === 'bubble' && !selectedChartCol3;

    if (!fileId || col1Missing || col2Missing || col3Missing) {
      alert("Пожалуйста, выберите все необходимые параметры для графика.");
      return;
    }

    setIsChartLoading(true);
    setChartData(null);
    setError(null);

    const formData = new FormData();
    formData.append('file_id', fileId);
    formData.append('chart_type', chartType);
    formData.append('column1', selectedChartCol1);
    if (selectedChartCol2) formData.append('column2', selectedChartCol2);
    if (selectedChartCol3) formData.append('column3', selectedChartCol3);

    try {
      const res = await api.post("/chart-data/", formData);
      setChartData(res.data);
    } catch (err: any) {
      const detail = err.response?.data?.detail || "Проверьте типы данных.";
      setError(`Ошибка при построении графика. ${detail}`);
    } finally {
      setIsChartLoading(false);
    }
  };

  if (isPageLoading) {
    return (
      <div className="flex justify-center items-center h-96">
        <Loader className="w-8 h-8 animate-spin text-gray-500" />
        <p className="ml-4 text-lg text-gray-600">Загрузка данных...</p>
      </div>
    );
  }

  if (!fileId) {
    return (
      <div className="text-center py-16 px-6 max-w-2xl mx-auto">
        <BarChart3 className="w-16 h-16 mx-auto text-gray-300" />
        <h2 className="mt-4 text-2xl font-semibold text-gray-800">Выберите файл</h2>
        <p className="mt-2 text-gray-500">Перейдите на главную страницу и загрузите файл.</p>
      </div>
    );
  }

  if (error && columns.length === 0) {
    return <div className="text-center py-16 text-red-600">{error}</div>;
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-gradient-to-r from-teal-500 to-cyan-500 rounded-xl">
            <TrendingUp className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-gray-800">Визуализация данных</h2>
            <p className="text-gray-600">Выберите тип графика и столбцы</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4 p-4 bg-gray-50 rounded-lg mb-6">
          <select value={chartType} onChange={e => setChartType(e.target.value)} className="p-2 border-gray-300 border rounded-md shadow-sm">
            <option value="histogram">Гистограмма</option>
            <option value="pie">Круговая диаграмма</option>
            <option value="line">Линейный график</option>
            <option value="area">Диаграмма с областями</option>
            <option value="scatter">Диаграмма рассеяния</option>
            <option value="bubble">Пузырьковая диаграмма</option>
            <option value="boxplot">Ящик с усами</option>
          </select>

          <select value={selectedChartCol1} onChange={e => setSelectedChartCol1(e.target.value)} className="p-2 border-gray-300 border rounded-md shadow-sm">
            <option value="">{['scatter', 'line', 'area', 'bubble'].includes(chartType) ? 'Выберите X' : 'Выберите столбец'}</option>
            {columns.map(c => <option key={`col1-${c.column}`} value={c.column}>{c.column}</option>)}
          </select>

          {['scatter', 'line', 'area', 'bubble'].includes(chartType) && (
            <select value={selectedChartCol2} onChange={e => setSelectedChartCol2(e.target.value)} className="p-2 border-gray-300 border rounded-md shadow-sm">
              <option value="">Выберите Y</option>
              {columns.map(c => <option key={`col2-${c.column}`} value={c.column}>{c.column}</option>)}
            </select>
          )}

          {chartType === 'bubble' && (
            <select value={selectedChartCol3} onChange={e => setSelectedChartCol3(e.target.value)} className="p-2 border-gray-300 border rounded-md shadow-sm">
              <option value="">Размер</option>
              {columns.map(c => <option key={`col3-${c.column}`} value={c.column}>{c.column}</option>)}
            </select>
          )}

          <button onClick={handleGenerateChart} disabled={isChartLoading} className="px-5 py-2 bg-blue-600 text-white rounded-md disabled:bg-gray-400 font-semibold hover:bg-blue-700 transition-colors">
            {isChartLoading ? <Loader className="w-5 h-5 animate-spin" /> : 'Построить'}
          </button>
        </div>

        <div className="min-h-[500px] p-4 border border-gray-200 rounded-lg bg-white relative">
          {isChartLoading && <div className="absolute inset-0 flex justify-center items-center bg-white/50"><Loader className="w-8 h-8 animate-spin" /></div>}
          {!isChartLoading && error && <div className="flex justify-center items-center h-full text-red-600">{error}</div>}

          {!isChartLoading && !error && chartData && chartData.chart_type === 'histogram' && (
            <Bar options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { display: false },
                title: { display: true, text: `Распределение '${selectedChartCol1}'` }
              }
            }} data={{
              labels: chartData.data.labels,
              datasets: [{
                label: selectedChartCol1,
                data: chartData.data.values,
                backgroundColor: 'rgba(59, 130, 246, 0.6)',
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 1
              }]
            }} />
          )}

          {!isChartLoading && !error && chartData && chartData.chart_type === 'pie' && (
            <div className="w-full h-full flex justify-center items-center">
              <div className="max-w-[450px] max-h-[450px]">
                <Pie options={{ responsive: true, plugins: { title: { display: true, text: `Доли '${selectedChartCol1}'` } } }} data={{
                  labels: chartData.data.labels,
                  datasets: [{
                    label: selectedChartCol1,
                    data: chartData.data.values,
                    backgroundColor: ['#4ade80', '#60a5fa', '#facc15', '#f87171', '#c084fc', '#fb923c', '#818cf8']
                  }]
                }} />
              </div>
            </div>
          )}

          {!isChartLoading && !error && chartData && chartData.chart_type === 'scatter' && (
            <Scatter options={{
              responsive: true,
              plugins: { title: { display: true, text: `Зависимость '${selectedChartCol1}' от '${selectedChartCol2}'` } }
            }} data={{
              datasets: [{
                label: 'Точки',
                data: chartData.data.points.map((p: any) => ({ x: p[selectedChartCol1], y: p[selectedChartCol2] })),
                backgroundColor: 'rgba(59, 130, 246, 0.6)'
              }]
            }} />
          )}

          {!isChartLoading && !error && chartData && ['line', 'area'].includes(chartData.chart_type) && (
            <Line options={{
              responsive: true,
              plugins: { title: { display: true, text: `Тренд '${selectedChartCol2}' по '${selectedChartCol1}'` } }
            }} data={{
              labels: chartData.data.labels,
              datasets: [{
                label: selectedChartCol2,
                data: chartData.data.values,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                tension: 0.1,
                fill: chartData.chart_type === 'area'
              }]
            }} />
          )}

          {!isChartLoading && !error && chartData && chartData.chart_type === 'bubble' && (
            <Bubble options={{
              responsive: true,
              plugins: { title: { display: true, text: `Пузырьки по '${selectedChartCol1}', '${selectedChartCol2}', '${selectedChartCol3}'` } }
            }} data={{
              datasets: [{
                label: selectedChartCol3,
                data: chartData.data.points.map((p: any) => ({
                  x: p[selectedChartCol1],
                  y: p[selectedChartCol2],
                  r: p[selectedChartCol3] / 2
                })),
                backgroundColor: 'rgba(59, 130, 246, 0.6)'
              }]
            }} />
          )}

          {!isChartLoading && !error && chartData && chartData.chart_type === 'boxplot' && (
  <Chart
    type="boxplot"
    options={{
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: `Ящик с усами для '${selectedChartCol1}'`
        },
        legend: { display: false }
      }
    }}
    data={{
      labels: [selectedChartCol1],
      datasets: [{
        label: selectedChartCol1,
        data: [chartData.data], // Пример: [min, q1, median, q3, max]
        backgroundColor: 'rgba(192, 75, 192, 0.6)',
        borderColor: 'rgb(192, 75, 192)',
        borderWidth: 1,
        itemRadius: 0
      }]
    }}
  />
)}
        </div>
      </div>
    </div>
  );
};

export default ChartsPage;
