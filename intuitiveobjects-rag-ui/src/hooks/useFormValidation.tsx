import React, { useState } from "react";

interface FormErrors {
	[key: string]: string;
}

const useFormValidation = (initialValues: any, validate: any) => {
	const [values, setValues] = useState(initialValues);
	const [errors, setErrors] = useState<FormErrors>({});

	const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
		// eslint-disable-next-line no-unsafe-optional-chaining
		const { name, value, dataset } = e?.target;

		const regexPattern = dataset?.regex ? new RegExp(dataset.regex) : null;

		if (regexPattern) {
			console.log(regexPattern.test(value));
			if (regexPattern.test(value)) {
				setValues((prev: any) => ({ ...prev, [name]: value }));
			}
		} else {
			setValues((prev: any) => ({ ...prev, [name]: value }));
		}
		setErrors({});
	};

	const onValidate = (): boolean => {
		const newErrors = validate(values);
		setErrors(newErrors);
		return Object.keys(newErrors).length === 0;
	};

	const onReset = () => {
		setValues(initialValues);
	};

	const handleSubmit = (callback: () => void) => {
		if (onValidate()) {
			callback();
		}
	};

	const handleBlur = () => {};

	return {
		values,
		errors,
		handleChange,
		handleSubmit,
		onReset,
		setValues,
		handleBlur,
		touched: {},
	};
};

export default useFormValidation;
