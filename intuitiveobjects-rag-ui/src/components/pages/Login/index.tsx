// import { useNavigate } from "react-router-dom";
// import { Button } from "@/components/atoms/Button/Button";
// import { TextInput } from "@/components/atoms/TextInput";
// import { useAppDispatch } from "@/store/hooks";
// import useFormValidation from "@/hooks/useFormValidation";
// import { authUserLoginApi } from "@/services/authApi";
// import { validateUserLoginForm } from "@/hooks/vaidate";
// import { showError, showSuccess } from "@/utils/sweetAlert";
// const initialState = {
// 	email: "",
// 	password: "",
// };

// const Login = () => {
// 	const dispatch = useAppDispatch();
// 	const navigate = useNavigate();
// 	const loginForm = useFormValidation(initialState, validateUserLoginForm);

// 	const onSubmit = async () => {
// 		console.log(loginForm.values);
// 		const response = await dispatch(authUserLoginApi(loginForm.values));

// 		if (response.type === "auth/user/login/fulfilled") {
// 			navigate("/");
// 			showSuccess("Login successful");
// 		}
// 		if (response.type === "auth/user/login/rejected") {
// 			showError(response.payload.response.data.message);
// 		}
// 	};
// 	return (
// 		<div className="flex min-h-screen items-center justify-center bg-chat-bg">
// 			<div className="w-full max-w-md p-8 border border-gray-600 shadow-lg rounded-2xl">
// 				<h2 className="text-2xl font-bold text-center text-gray-400 mb-6">Welcome Back</h2>
// 				<form
// 					onSubmit={(e) => {
// 						e.preventDefault();
// 						console.log(loginForm.values);
// 						loginForm.handleSubmit(onSubmit);
// 					}}>
// 					<div className="mb-4">
// 						<TextInput
// 							label="Email Address"
// 							type="email"
// 							placeholder="Enter your email"
// 							name="email"
// 							onChange={loginForm.handleChange}
// 							error={loginForm.errors.email}
// 						/>
// 					</div>
// 					<div className="mb-6">
// 						<TextInput
// 							label="Password"
// 							type="password"
// 							placeholder="Enter your password"
// 							name="password"
// 							onChange={loginForm.handleChange}
// 							error={loginForm.errors.password}
// 						/>
// 					</div>
// 					<Button className="w-full" variant="primary" type="submit">
// 						Log In
// 					</Button>
// 				</form>
// 			</div>
// 		</div>
// 	);
// };

// export default Login;
import { useNavigate } from "react-router-dom";
import { useAppDispatch } from "@/store/hooks";
import { Button } from "@/components/atoms/Button/Button";
import { TextInput } from "@/components/atoms/TextInput";
import useFormValidation from "@/hooks/useFormValidation";
import { authUserLoginApi } from "@/services/authApi";
import { validateUserLoginForm } from "@/hooks/vaidate";
import { showError, showSuccess } from "@/utils/sweetAlert";
import { FiSettings } from "react-icons/fi";
import { useState } from "react"; // <--- make sure this is added

const initialState = {
  email: "",
  password: "",
};

const Login = () => {
  const dispatch = useAppDispatch();
  const navigate = useNavigate();
  const loginForm = useFormValidation(initialState, validateUserLoginForm);
  const [open, setOpen] = useState(false); // <-- for settings dropdown

  const onSubmit = async () => {
    const response = await dispatch(authUserLoginApi(loginForm.values));

    if (response.type === "auth/user/login/fulfilled") {
      navigate("/");
      showSuccess("Login successful");
    }
    if (response.type === "auth/user/login/rejected") {
      showError(response.payload.response.data.message);
    }
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center bg-chat-bg">
      {/* 👇 Settings Icon Top-Right */}
      <div className="absolute top-4 right-4 z-10">
        <button
          onClick={() => setOpen(!open)}
          className="text-white hover:text-gray-300"
          aria-label="Settings"
        >
          <FiSettings size={24} />
        </button>

        {open && (
          <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg">
            <button
              onClick={() => navigate("/admin")}
              className="block w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100"
            >
              Login as Admin
            </button>
          </div>
        )}
      </div>

      {/* Login Card */}
      <div className="w-full max-w-md p-8 border border-gray-600 shadow-lg rounded-2xl">
        <h2 className="text-2xl font-bold text-center text-gray-400 mb-6">
          Welcome Back
        </h2>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            loginForm.handleSubmit(onSubmit);
          }}
        >
          <div className="mb-4">
            <TextInput
              label="Email Address"
              type="email"
              placeholder="Enter your email"
              name="email"
              onChange={loginForm.handleChange}
              error={loginForm.errors.email}
            />
          </div>
          <div className="mb-6">
            <TextInput
              label="Password"
              type="password"
              placeholder="Enter your password"
              name="password"
              onChange={loginForm.handleChange}
              error={loginForm.errors.password}
            />
          </div>
          <Button className="w-full" variant="primary" type="submit">
            Log In
          </Button>
        </form>
      </div>
    </div>
  );
};

export default Login;
