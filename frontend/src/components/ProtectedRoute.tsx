import React, { useContext } from 'react';
import { Navigate, useLocation, Outlet } from 'react-router-dom';
import { AppContext } from '../contexts/AppContext';

const ProtectedRoute = () => {
  const { token, isLoading } = useContext(AppContext)!;
  const location = useLocation();
  if (isLoading) {
    return <div>Проверка авторизации...</div>;
  }
  if (token) {
    return <Outlet />;
  }
  return <Navigate to="/login" state={{ from: location }} replace />;
};

export default ProtectedRoute;