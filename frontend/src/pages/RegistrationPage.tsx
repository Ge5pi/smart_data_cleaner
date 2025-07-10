import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api';
import { UserPlus, Mail, KeyRound, Loader } from 'lucide-react';

const RegistrationPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordConfirm, setPasswordConfirm] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (password !== passwordConfirm) {
      setError('Пароли не совпадают.');
      return;
    }
    if (!email || !password) {
        setError('Пожалуйста, заполните все поля.');
        return;
    }

    setIsLoading(true);
    setError(null);

    // --- НАЧАЛО ИСПРАВЛЕНИЙ ---

    // 1. Создаем обычный JavaScript объект вместо FormData.
    //    Используем ключ 'email', как того требует Pydantic-схема на бэкенде.
    const userData = {
      email: email,
      password: password
    };

    try {
      // 2. Передаем объект userData напрямую.
      //    Axios автоматически преобразует его в JSON и установит правильный заголовок Content-Type.
      await api.post('/users/register', userData);

      alert('Регистрация прошла успешно! Теперь вы можете войти.');
      // Примечание: Убедитесь, что вы используете правильный порт (8000, а не 5643)
      navigate('/login'); // Перенаправляем на страницу входа

    // --- КОНЕЦ ИСПРАВЛЕНИЙ ---

    } catch (err: any) {
      if (err.response && err.response.data && err.response.data.detail) {
        setError(err.response.data.detail);
      } else {
        setError('Произошла ошибка при регистрации.');
      }
      console.error(err);
    } finally {
      setIsLoading(false); // Этот блок выполнится в любом случае
    }
};

  return (
    <div className="max-w-md mx-auto px-6 py-12">
      <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-8">
        <div className="flex flex-col items-center mb-6">
          <div className="p-3 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-xl mb-4">
            <UserPlus className="w-8 h-8 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-800">Создать аккаунт</h2>
          <p className="text-gray-600 mt-1">Присоединяйтесь к нашему сервису</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="relative">
            <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full pl-10 pr-4 py-3 rounded-xl border-2 border-gray-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition"
              required
            />
          </div>
          <div className="relative">
            <KeyRound className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="password"
              placeholder="Пароль"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full pl-10 pr-4 py-3 rounded-xl border-2 border-gray-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition"
              required
            />
          </div>
          <div className="relative">
             <KeyRound className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="password"
              placeholder="Подтвердите пароль"
              value={passwordConfirm}
              onChange={(e) => setPasswordConfirm(e.target.value)}
              className="w-full pl-10 pr-4 py-3 rounded-xl border-2 border-gray-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition"
              required
            />
          </div>

          {error && <p className="text-sm text-center text-red-600 bg-red-100 p-3 rounded-lg">{error}</p>}

          <button
            type="submit"
            disabled={isLoading}
            className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-semibold hover:shadow-lg disabled:opacity-70 transition"
          >
            {isLoading ? <Loader className="w-5 h-5 animate-spin" /> : 'Зарегистрироваться'}
          </button>
          <div className="relative">
          <p className="text-gray-600 px-1  font-semibold gap-2 inline text-lg">Уже есть аккаунт?</p> <a href="/login"> <h3 className="text-lg gap-2 text-blue-500 inline font-bold">Войти</h3></a>
          </div>
        </form>
      </div>
    </div>
  );
};

export default RegistrationPage;