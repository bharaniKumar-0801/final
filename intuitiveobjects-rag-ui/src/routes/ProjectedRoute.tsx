import { Outlet } from "react-router-dom";
import { useAppSelector } from "@/store/hooks";
import { Navigate } from "react-router-dom";

export const AdminProtectedRoute = () => {
  const isAuthenticated = useAppSelector(
    (state) => state.auth.admin?.isAuthenticated
  );

  // If not authenticated, redirect to admin login
  if (!isAuthenticated) {
    return <Navigate to="/admin/login" replace />;
  }

  // If authenticated, render the child routes
  return <Outlet />;
};

export const AdminAuthRoute = () => {
  const isAuthenticated = useAppSelector(
    (state) => state.auth.admin?.isAuthenticated
  );

  // If already authenticated, redirect to admin dashboard
  if (isAuthenticated) {
    return <Navigate to="/admin" replace />;
  }

  // If not authenticated, show the login page
  return <Outlet />;
};

export const UserProtectedRoute = () => {
  const isAuthenticated = useAppSelector(
    (state) => state.auth.user?.isAuthenticated
  );

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <Outlet />;
};

export const UserAuthRoute = () => {
  const isAuthenticated = useAppSelector(
    (state) => state.auth.user?.isAuthenticated
  );

  if (isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  return <Outlet />;
};
