import { useContext } from 'react';
import { Navigate, useLocation, Outlet } from 'react-router-dom';
import { AppContext } from '../contexts/AppContext';

const ProtectedRoute = () => {
  const { token, isAuthCheckComplete } = useContext(AppContext)!;
  const location = useLocation();
  if (!isAuthCheckComplete) {
    return <Outlet />;
  }
  if (!token) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  return <Outlet />;
};

export default ProtectedRoute;