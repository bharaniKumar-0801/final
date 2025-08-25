import React from "react";
import ReactDOM from "react-dom/client";
import { Provider } from "react-redux";
import makeStore from "@/store/store";
import "./styles/index.scss";
import { PersistGate } from "redux-persist/integration/react";
import { Toaster } from "react-hot-toast";
import { BrowserRouter } from "react-router-dom";
import App from "./app";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Provider store={makeStore.reduxStore}>
        <PersistGate loading={null} persistor={makeStore.persistor}>
          <App />
          <Toaster />
        </PersistGate>
      </Provider>
    </BrowserRouter>
  </React.StrictMode>
);
