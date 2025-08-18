// import { useState } from "react";
// import { useNavigate } from "react-router-dom";
// import { TextInput } from "@/components/atoms/TextInput";
// import { Button } from "@/components/atoms/Button/Button";
// import { useAppDispatch } from "@/store/hooks";
// import { orgAdminLoginApi } from "@/services/adminApi";
// import useFormValidation from "@/hooks/useFormValidation";
// import { validateAdminLoginForm } from "@/hooks/vaidate";
// import toast from "react-hot-toast";

// const initialLoginFormState = {
//   email: "",
//   password: "",
// };

// export const AdminLogin = () => {
//   const dispatch = useAppDispatch();
//   const navigate = useNavigate();

// 	const adminLoginForm = useFormValidation(initialLoginFormState, validateAdminLoginForm);
// 	const [loading, setLoading] = useState(false);

// 	const onSubmit = async () => {
// 		setLoading(true);
// 		try {
// 			const response = await dispatch(orgAdminLoginApi(adminLoginForm.values));
// 			if (response.type === "orgAdmin/login/fulfilled") {
// 				navigate("/admin");
// 				toast.success("Successfully logged in!", {
// 					duration: 3000,
// 					position: "top-right",
// 				});
// 			} else {
// 				toast.error(response.payload.response.data.message || "Login failed. Please try again.", {
// 					duration: 3000,
// 					position: "top-right",
// 				});
// 			}
// 		} catch (err: any) {
// 			console.log(err);
// 		} finally {
// 			setLoading(false);
// 		}
// 	};

//   return (
//     <div className="min-h-screen flex items-center justify-center bg-chat-bg">
//       <div className="max-w-md w-full rounded-2xl border border-gray-600 shadow-md p-8">
//         <h1 className="text-2xl text-center font-bold text-gray-400 mb-6">
//           Admin Login
//         </h1>

// 				{/* {error && <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">{error}</div>} */}

// 				<form
// 					onSubmit={(e) => {
// 						e.preventDefault();
// 						adminLoginForm.handleSubmit(onSubmit);
// 					}}>
// 					<div className="mb-4">
// 						<TextInput
// 							label="Email"
// 							type="email"
// 							name="email"
// 							value={adminLoginForm.values.email}
// 							onChange={adminLoginForm.handleChange}
// 							placeholder="Enter your email"
// 							error={adminLoginForm.errors.email}
// 						/>
// 					</div>

//           <div className="mb-6">
//             <TextInput
//               label="Password"
//               type="password"
//               name="password"
//               value={adminLoginForm.values.password}
//               onChange={adminLoginForm.handleChange}
//               error={adminLoginForm.errors.password}
//               placeholder="Enter your password"
//             />
//           </div>

//           <Button type="submit" disabled={loading} className="w-full">
//             {loading ? "Logging in..." : "Login"}
//           </Button>
//         </form>
//       </div>
//     </div>
//   );
// };


import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { TextInput } from "@/components/atoms/TextInput";
import { Button } from "@/components/atoms/Button/Button";
import { useAppDispatch } from "@/store/hooks";
import {
  orgAdminLoginApi,
  registerOrganizationAdminApi,
  registerOrganizationAdminApiWithEmail,
  // sendPasswordResetEmailApi,
  sendOtpApi,
  verifyOtpAndUpdateAdminApi,
} from "@/services/adminApi";
import useFormValidation from "@/hooks/useFormValidation";
import { validateAdminLoginForm } from "@/hooks/vaidate";
import toast from "react-hot-toast";
// Utility to generate random password
function generateRandomPassword(length = 12) {
  const chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()";
  let password = "";
  for (let i = 0; i < length; i++) {
    password += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return password;
}
const initialLoginFormState = { email: "", password: "" };
const initialSignUpFormState = { organizationName: "", email: "" };
export const AdminLogin = () => {
  const dispatch = useAppDispatch();
  const navigate = useNavigate();
  const [isSignUp, setIsSignUp] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showChangeCredentials, setShowChangeCredentials] = useState(false);
  const [changeForm, setChangeForm] = useState({
    organizationName: "",
    email: "",
  });
  const [otpSent, setOtpSent] = useState(false);
  const [otp, setOtp] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const adminLoginForm = useFormValidation(
    initialLoginFormState,
    validateAdminLoginForm
  );
  const [signUpForm, setSignUpForm] = useState(initialSignUpFormState);
  const handleSignUpChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setSignUpForm((prev) => ({ ...prev, [name]: value }));
  };
  const handleChangeForm = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setChangeForm((prev) => ({ ...prev, [name]: value }));
  };
  const handleSignUp = async () => {
    if (!signUpForm.email || !signUpForm.organizationName) {
      toast.error("Please fill all fields");
      return;
    }
    setLoading(true);
    let res: any = null;
    try {
      res = await dispatch(
        registerOrganizationAdminApi({
          name: signUpForm.organizationName,
        })
      );
      if (res.type === "organization/register/fulfilled") {
        toast.success("Organization registered successfully.");
        setIsSignUp(false);
        setSignUpForm(initialSignUpFormState);
      } else {
        toast.error("Registration failed");
      }
    } catch (err) {
      console.error(err);
      toast.error("Error occurred while registering.");
    } finally {
      setLoading(false);
    }
    if (res && res.payload && res.payload.data && res.payload.data.id) {
      setLoading(true);
      try {
        const response = await dispatch(
          registerOrganizationAdminApiWithEmail({
            organization_id: res.payload.data.id,
            name: signUpForm.organizationName,
            email: signUpForm.email,
            password: "epr_admin@123", // Default password
          })
        );
        if (
          response.type === "organization-admin/organizationadmin/fulfilled"
        ) {
          toast.success("Admin registered successfully!");
          setIsSignUp(false);
          setSignUpForm(initialSignUpFormState);
        } else {
          toast.error("Registration failed");
        }
      } catch (err) {
        console.error(err);
        toast.error("Error occurred while registering with email.");
      } finally {
        setLoading(false);
      }
    }
  };
  // Step 1: Request OTP for change credentials
  const handleRequestOtp = async () => {
    if (!changeForm.organizationName || !changeForm.email) {
      toast.error("Please fill all fields");
      return;
    }
    setLoading(true);
    try {
      const res = await dispatch(
        sendOtpApi({
          organizationName: changeForm.organizationName,
          email: changeForm.email,
        })
      );
      if (res.type === "admin/sendOtp/fulfilled") {
        toast.success("OTP sent to your email.");
        setOtpSent(true);
      } else {
        toast.error("Failed to send OTP.");
      }
    } catch {
      toast.error("Error sending OTP.");
    } finally {
      setLoading(false);
    }
  };
  // Step 2: Verify OTP and update credentials
  const handleVerifyOtpAndUpdate = async () => {
    if (!otp || !newPassword || !newEmail) {
      toast.error("Please fill all fields");
      return;
    }
    setLoading(true);
    try {
      const res = await dispatch(
        verifyOtpAndUpdateAdminApi({
          organizationName: changeForm.organizationName,
          email: changeForm.email,
          otp,
          newPassword,
          newEmail: newEmail || changeForm.email, // Use newEmail if provided, otherwise keep current email
        })
      );
      if (res.type === "admin/verifyOtpAndUpdate/fulfilled") {
        toast.success("Credentials updated successfully!");
        setShowChangeCredentials(false);
        setOtpSent(false);
        setChangeForm({ organizationName: "", email: "" });
        setOtp("");
        setNewPassword("");
        setNewEmail("");
      } else {
        toast.error("Failed to update credentials.");
      }
    } catch {
      toast.error("Error updating credentials.");
    } finally {
      setLoading(false);
    }
  };
  // Login
  const onSubmit = async () => {
    setLoading(true);
    try {
      const response = await dispatch(orgAdminLoginApi(adminLoginForm.values));
      if (response.type === "orgAdmin/login/fulfilled") {
        navigate("/admin");
        toast.success("Successfully logged in!");
      } else {
        toast.error(
          response.payload?.response?.data?.message || "Login failed"
        );
      }
    } catch (err) {
      console.log(err);
    } finally {
      setLoading(false);
    }
  };
return (
    <div className="min-h-screen flex items-center justify-center bg-chat-bg">
      <div className="max-w-md w-full rounded-2xl border border-gray-600 shadow-md p-8">
        <h1 className="text-2xl text-center font-bold text-gray-400 mb-6">
          {isSignUp
            ? "Admin Sign Up"
            : showChangeCredentials
            ? "Change Email/Password"
            : "Admin Login"}
        </h1>
        {!isSignUp && !showChangeCredentials ? (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              onSubmit();
            }}
          >
            <TextInput
              label="Email"
              type="email"
              name="email"
              value={adminLoginForm.values.email}
              onChange={adminLoginForm.handleChange}
              placeholder="Enter your email"
              error={adminLoginForm.errors.email}
            />
            <TextInput
              label="Password"
              type="password"
              name="password"
              value={adminLoginForm.values.password}
              onChange={adminLoginForm.handleChange}
              error={adminLoginForm.errors.password}
              placeholder="Enter your password"
              className="mt-4"
            />
            <Button type="submit" disabled={loading} className="w-full mt-6">
              {loading ? "Logging in..." : "Login"}
            </Button>
            <div className="text-sm text-gray-400 mt-4 text-center">
              <p>
                New Admin?{" "}
                <button
                  type="button"
                  onClick={() => setIsSignUp(true)}
                  className="text-blue-500 underline"
                >
                  Sign up here
                </button>
              </p>
              <button
                type="button"
                onClick={() => setShowChangeCredentials(true)}
                className="text-blue-500 underline text-sm ml-2"
              >
                Change Email/Password?
              </button>
            </div>
          </form>
        ) : showChangeCredentials ? (
          <div>
            {!otpSent ? (
              <>
                <TextInput
                  label="Organization Name"
                  type="text"
                  name="organizationName"
                  value={changeForm.organizationName}
                  onChange={handleChangeForm}
                  placeholder="Enter organization name"
                />
                <TextInput
                  label="Current Email"
                  type="email"
                  name="email"
                  value={changeForm.email}
                  onChange={handleChangeForm}
                  placeholder="Enter your current email"
                  className="mt-4"
                />
                <Button
                  onClick={handleRequestOtp}
                  disabled={loading}
                  className="w-full mt-6"
                >
                  {loading ? "Sending OTP..." : "Send OTP"}
                </Button>
                <p className="text-sm text-gray-400 mt-4 text-center">
                  <button
                    type="button"
                    onClick={() => setShowChangeCredentials(false)}
                    className="text-blue-500 underline"
                  >
                    Back to Login
                  </button>
                </p>
              </>
            ) : (
              <>
                <TextInput
                  label="OTP"
                  type="text"
                  name="otp"
                  value={otp}
                  onChange={(e) => setOtp(e.target.value)}
                  placeholder="Enter OTP"
                />
                <TextInput
                  label="New Email (optional)"
                  type="email"
                  name="newEmail"
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
                  placeholder="Enter new email (optional)"
                  className="mt-4"
                />
                <TextInput
                  label="New Password"
                  type="password"
                  name="newPassword"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="Enter new password"
                  className="mt-4"
                />
                <Button
                  onClick={handleVerifyOtpAndUpdate}
                  disabled={loading}
                  className="w-full mt-6"
                >
                  {loading ? "Updating..." : "Update Credentials"}
                </Button>
                <p className="text-sm text-gray-400 mt-4 text-center">
                  <button
                    type="button"
                    onClick={() => {
                      setOtpSent(false);
                      setOtp("");
                    }}
                    className="text-blue-500 underline"
                  >
                    Back
                  </button>
                </p>
              </>
            )}
          </div>
        ) : (
          <div>
            <TextInput
              label="Organization Name"
              type="text"
              name="organizationName"
              value={signUpForm.organizationName}
              onChange={handleSignUpChange}
              placeholder="Enter organization name"
            />
            <TextInput
              label="Email"
              type="email"
              name="email"
              value={signUpForm.email}
              onChange={handleSignUpChange}
              placeholder="Enter your email"
              className="mt-4"
            />
            <Button
              onClick={handleSignUp}
              disabled={loading}
              className="w-full mt-6"
            >
              {loading ? "Signing up..." : "Sign Up"}
            </Button>
            <p className="text-sm text-gray-400 mt-4 text-center">
              Already have an account?{" "}
              <button
                type="button"
                onClick={() => setIsSignUp(false)}
                className="text-blue-500 underline"
              >
                Login here
              </button>
            </p>
          </div>
        )}
      </div>
    </div>
  );
};