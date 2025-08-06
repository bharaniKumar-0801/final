interface AdminLoginFormParams {
    email?: string;
    password?: string;
}

export const validateAdminLoginForm = (values: AdminLoginFormParams) => {
    const errors: AdminLoginFormParams = {};
  
    if (!values.email) errors.email = "Email is required";
    else if (!/\S+@\S+\.\S+/.test(values.email)) {
      errors.email = "Email is invalid";
    }
    if (!values.password) errors.password = "Password is required";
    else if (values.password.length < 6) {
      errors.password = "Password must be at least 6 characters";
    }
  
    return errors;
  };
  
  interface CategoryFormParams {
    name?: string;
  }

  export const validateCategoryForm = (values: CategoryFormParams) => {
    const errors: CategoryFormParams = {};

    if (!values.name) errors.name = "Category name is required";

    return errors;
  };

  interface UserFormParams {
    email?: string;
    password?: string;
    category_id?: string;
  }

  export const validateUserForm = (values: UserFormParams) => {
    const errors: UserFormParams = {};

    if (!values.email) errors.email = "Email is required";
    else if (!/\S+@\S+\.\S+/.test(values.email)) {
      errors.email = "Email is invalid";
    }

    if (!values.password) errors.password = "Password is required";
    else if (values.password.length < 6) {
      errors.password = "Password must be at least 6 characters";
    }

    if (!values.category_id) errors.category_id = "Category is required";

    return errors;
  };

  interface IngestionFormParams {
    category_id?: string;
    file?: File | null;
    tags?: string[];
  }
  
  export const validateIngestionForm = (values: IngestionFormParams) => {
    const errors: any = {};
    if (!values.category_id) errors.category_id = "Category is required";
    if (!values.file) errors.file = "File is required" as any;
    if (!values.tags || values.tags.length === 0) {
        errors.tags = "Tags are required";
    }

    return errors;
  };
  

  interface UserLoginFormParams {
    email?: string;
    password?: string;
  }

  export const validateUserLoginForm = (values: UserLoginFormParams) => {
    const errors: UserLoginFormParams = {};

    if (!values.email) errors.email = "Email is required";
    else if (!/\S+@\S+\.\S+/.test(values.email)) {
      errors.email = "Email is invalid";
    }

    if (!values.password) errors.password = "Password is required";
    else if (values.password.length < 6) {
      errors.password = "Password must be at least 6 characters";
    }

    return errors;
  };
