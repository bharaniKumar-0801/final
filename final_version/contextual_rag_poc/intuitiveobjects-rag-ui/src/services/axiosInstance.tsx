import axios from "axios";

import makeStore from "@/store/store";

const axiosAdminInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
});

axiosAdminInstance.interceptors.request.use((config) => {
  const token = makeStore.reduxStore.getState().auth.admin?.token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

const axiosUserInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
});

axiosUserInstance.interceptors.request.use((config) => {
  const token = makeStore.reduxStore.getState().auth.user?.token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export { axiosAdminInstance, axiosUserInstance };