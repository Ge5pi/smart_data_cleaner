import React, { createContext, useState, ReactNode } from 'react';

// Определим типы для полноты картины
type ColumnAnalysis = {
  column: string;
  dtype: string;
  nulls: number;
  unique: number;
  sample_values: string[];
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
};

export const AppContext = createContext<AppContextType | null>(null);

export const AppProvider = ({ children }: { children: ReactNode }) => {
  const [file, setFile] = useState<File | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const [columns, setColumns] = useState<ColumnAnalysis[]>([]);
  const [preview, setPreview] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

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
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};