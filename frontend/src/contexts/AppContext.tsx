import { createContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import { jwtDecode } from 'jwt-decode';

type ColumnAnalysis = {
  column: string;
  dtype: string;
  nulls: number;
  unique: number;
  sample_values: string[];
};

type User = {
    email: string;
};



type AppContextType = {
  file: File | null;
  setFile: React.Dispatch<React.SetStateAction<File | null>>;
  fileId: string | null;
  setFileId: React.Dispatch<React.SetStateAction<string | null>>;
  sessionId: string | null;
  setSessionId: React.Dispatch<React.SetStateAction<string | null>>;
  columns: ColumnAnalysis[];
  setColumns: React.Dispatch<React.SetStateAction<ColumnAnalysis[]>>;
  preview: any[];
  setPreview: React.Dispatch<React.SetStateAction<any[]>>;
  error: string | null;
  setError: React.Dispatch<React.SetStateAction<string | null>>;
  isLoading: boolean;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
  resetState: () => void;
  token: string | null;
  user: User | null;
  login: (token: string) => void;
  logout: () => void;
  userFiles: any[];
  setUserFiles: React.Dispatch<React.SetStateAction<any[]>>;
  isAuthCheckComplete: boolean;
};

export const AppContext = createContext<AppContextType | null>(null);

export const AppProvider = ({ children }: { children: ReactNode }) => {
  const [file, setFile] = useState<File | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isAuthCheckComplete, setIsAuthCheckComplete] = useState(false);
  const [columns, setColumns] = useState<ColumnAnalysis[]>([]);
  const [preview, setPreview] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [token, setToken] = useState<string | null>(localStorage.getItem('authToken'));
  const [user, setUser] = useState<User | null>(null);
  const [userFiles, setUserFiles] = useState<any[]>([]);

  useEffect(() => {
        const storedToken = localStorage.getItem('authToken');
        if (storedToken) {
            try {
                const decoded: any = jwtDecode(storedToken);
                // Проверяем срок годности токена
                if (decoded.exp * 1000 > Date.now()) {
                    setToken(storedToken);
                    setUser({ email: decoded.sub });
                } else {
                    localStorage.removeItem('authToken'); // Удаляем просроченный токен
                }
            } catch (error) {
                console.error("Invalid token found in localStorage", error);
                localStorage.removeItem('authToken');
            }
        }
        setIsAuthCheckComplete(true);
  }, []);

  const login = (newToken: string) => {
        localStorage.setItem('authToken', newToken);
        setToken(newToken);
        try {
            const decoded: any = jwtDecode(newToken);
            setUser({ email: decoded.sub });
        } catch (error) {
            console.error("Failed to decode token", error);
        }
  };

  const logout = () => {
        localStorage.removeItem('authToken');
        setToken(null);
        setUser(null);
        setFileId(null);
        setSessionId(null);
        setColumns([]);
        setPreview([]);
    };

  // Функция сброса состояния при выборе нового файла
  const resetState = () => {
    setFile(null);
    setFileId(null);
    setSessionId(null);
    setColumns([]);
    setPreview([]);
    setError(null);
  };

  const value = {
    file, setFile,
    fileId, setFileId,
    sessionId, setSessionId,
    columns, setColumns,
    preview, setPreview,
    error, setError,
    isLoading, setIsLoading,
    resetState,
    token,
    user,
    login,
    logout,
    isAuthCheckComplete,
    userFiles,
    setUserFiles
  };
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};