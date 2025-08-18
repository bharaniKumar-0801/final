import Swal from "sweetalert2";

export const confirmAction = (title: string, text: string, confirmButtonText: string = "Yes", icon: "warning" | "question" | "info" = "warning") => {
	return Swal.fire({
		title,
		text,
		icon,
		showCancelButton: true,
		confirmButtonColor: "#3085d6",
		cancelButtonColor: "#d33",
		confirmButtonText,
	});
};

export const showSuccess = (title: string, text?: string) => {
	return Swal.fire({
		title,
		text,
		icon: "success",
		timer: 2000,
		timerProgressBar: true,
	});
};

export const showError = (title: string, text?: string) => {
	return Swal.fire({
		title,
		text,
		icon: "error",
	});
};
