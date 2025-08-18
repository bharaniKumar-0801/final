import { useRoutes } from "react-router-dom";
import { Home } from "@/components/pages/Home";
import { Admin } from "@/components/pages/Admin";
import { AdminLogin } from "@/components/pages/Admin/Login";
import {
  AdminProtectedRoute,
  AdminAuthRoute,
  UserProtectedRoute,
  UserAuthRoute,
} from "@/routes/ProjectedRoute";
import Login from "@/components/pages/Login";
import { AdminLayout } from "@/components/template/AdminLayout/AdminLayout";
const Routes = () => {
  return useRoutes([
    {
      element: <UserProtectedRoute />,
      children: [
        {
          path: "/",
          element: <Home />,
        },
        {
          path: "/chat/:chatId",
          element: <Home />,
        },
      ],
    },
    {
      element: <UserAuthRoute />,
      children: [
        {
          path: "/login",
          element: <Login />,
        },
      ],
    },
    {
      element: <AdminAuthRoute />,
      children: [
        {
          path: "/admin/login",
          element: <AdminLogin />,
        },
      ],
    },
    {
      element: <AdminProtectedRoute />,
      children: [
        {
          element: <AdminLayout />,
          children: [
            {
              path: "/admin",
              element: <Admin />,
            },
          ],
        },
      ],
    },
  ]);
};

export default Routes;
