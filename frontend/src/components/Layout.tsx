import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { Database, BotMessageSquare, LogIn, UserPlus, LogOut, BarChartHorizontal } from 'lucide-react';
import { useContext } from 'react';
import { AppContext } from '../contexts/AppContext';

const Layout = () => {
  const { user, logout, fileId } = useContext(AppContext)!;
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 font-sans">
      <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200/50 sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          {/* Левая часть: Логотип и Навигация */}
          <div className="flex items-center gap-10">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl">
                <Database className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
                Smart Data Tool
              </h1>
            </div>
            <nav className="flex items-center gap-2">
              <NavLink to="/" className={({ isActive }) => `px-4 py-2 rounded-lg text-sm font-medium transition-colors ${isActive ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}>
                Инструменты
              </NavLink>

              {/* УЛУЧШЕНИЕ: Ссылка на чат также должна быть неактивна без fileId */}
              <NavLink
                to="/chat"
                className={({ isActive }) => `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${isActive ? 'bg-indigo-100 text-indigo-700' : 'text-gray-600 hover:bg-gray-100'} ${!fileId ? 'opacity-50 cursor-not-allowed' : ''}`}
                onClick={(e) => !fileId && e.preventDefault()}
              >
                <BotMessageSquare className="w-4 h-4" />
                AI Агент
              </NavLink>

              <NavLink
                to="/charts"
                className={({ isActive }) => `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${isActive ? 'bg-teal-100 text-teal-700' : 'text-gray-600 hover:bg-gray-100'} ${!fileId ? 'opacity-50 cursor-not-allowed' : ''}`}
                onClick={(e) => !fileId && e.preventDefault()}
              >
                <BarChartHorizontal className="w-4 h-4" />
                Графики
              </NavLink>
            </nav>
          </div>

          {/* Правая часть: Статус пользователя */}
          <div className="flex items-center gap-4">
            {user ? (
              <>
                <span className="text-sm text-gray-700">
                  Вы вошли как: <span className="font-semibold">{user.email}</span>
                </span>
                <button
                  onClick={handleLogout}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-red-600 hover:bg-red-100 transition-colors"
                >
                  <LogOut className="w-4 h-4" />
                  Выйти
                </button>
              </>
            ) : (
              <>
                <NavLink to="/login" className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-gray-600 hover:bg-gray-100 transition-colors">
                  <LogIn className="w-4 h-4" />
                  Войти
                </NavLink>
                <NavLink to="/register" className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-blue-600 text-white hover:bg-blue-700 transition-colors">
                  <UserPlus className="w-4 h-4" />
                  Регистрация
                </NavLink>
              </>
            )}
          </div>
        </div>
      </div>
      <main>
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;