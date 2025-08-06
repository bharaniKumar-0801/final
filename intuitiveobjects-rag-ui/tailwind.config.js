/** @type {import('tailwindcss').Config} */
export default {
	content: [
		"./src/components/**/*.{js,jsx,ts,tsx}",
		"./src/atoms/**/*.{js,jsx,ts,tsx}",
		"./src/molecules/**/*.{js,jsx,ts,tsx}",
		"./src/organisms/**/*.{js,jsx,ts,tsx}",
		"./src/templates/**/*.{js,jsx,ts,tsx}",
		"./src/pages/**/*.{js,jsx,ts,tsx}",
		"./index.html",
	],
	theme: {
		extend: {
			colors: {
				"chat-bg": "#343541",
				"sidebar-bg": "#343541",
				"chat-border": "#4a4b53",
				"user-message": "#343541",
				"assistant-message": "#343541",
				"hover-bg": "#2A2B32",
			},
		},
	},
	plugins: [],
};
