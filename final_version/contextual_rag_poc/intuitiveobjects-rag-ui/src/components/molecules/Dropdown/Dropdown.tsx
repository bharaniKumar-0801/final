import { useRef, useEffect } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useAppDispatch } from "@/store/hooks";
import { logout } from "@/store/slices/userSlice";

interface DropdownProps {
	isOpen: boolean;
	onClose: () => void;
}

export const Dropdown = ({ isOpen, onClose }: DropdownProps) => {
	const dropdownRef = useRef<HTMLDivElement>(null);
	const dispatch = useAppDispatch();

	useEffect(() => {
		const handleClickOutside = (event: MouseEvent) => {
			if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
				onClose();
			}
		};

		if (isOpen) {
			document.addEventListener("mousedown", handleClickOutside);
		}

		return () => {
			document.removeEventListener("mousedown", handleClickOutside);
		};
	}, [isOpen, onClose]);

	if (!isOpen) return null;

	return (
		<div ref={dropdownRef} className="absolute right-0 top-12 w-64 bg-sidebar-bg rounded-lg shadow-lg border border-chat-border">
			<div className="py-2">
				<button
					onClick={() => dispatch(logout())}
					className="w-full px-4 py-2 text-left text-white hover:bg-hover-bg flex items-center gap-3">
					<Icon name="plus" className="w-5 h-5" />
					Log out
				</button>
			</div>
		</div>
	);
};
