import React from "react";

interface Column {
	key: string;
	header: string;
	render?: (value: any, row: any) => React.ReactNode;
	className?: string;
}

interface ResponsiveTableProps {
	columns: Column[];
	data: any[];
	emptyMessage?: string;
	className?: string;
	isLoading?: boolean;
}

export const ResponsiveTable: React.FC<ResponsiveTableProps> = ({
	columns,
	data,
	emptyMessage = "No data available",
	className = "",
	isLoading = false,
}) => {
	if (isLoading) {
		return (
			<div className="py-8 text-center text-gray-400">
				<svg className="animate-spin h-8 w-8 mx-auto text-gray-400 mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
					<circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
					<path
						className="opacity-75"
						fill="currentColor"
						d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
				</svg>
				<p>Loading...</p>
			</div>
		);
	}

	if (!data || data.length === 0) {
		return <div className="text-center text-gray-400 py-8">{emptyMessage}</div>;
	}

	return (
		<div className={`overflow-hidden rounded-lg shadow ${className}`}>
			{/* Desktop Table View */}
			<div className="hidden md:block overflow-x-auto">
				<table className="min-w-full divide-y divide-gray-700">
					<thead className="bg-primary-100">
						<tr>
							{columns.map((column) => (
								<th
									key={column.key}
									scope="col"
									className={`px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider ${
										column.className || ""
									}`}>
									{column.header}
								</th>
							))}
						</tr>
					</thead>
					<tbody className="divide-y divide-gray-700">
						{data.map((row, rowIndex) => (
							<tr key={rowIndex} className="hover:bg-gray-600">
								{columns.map((column) => (
									<td
										key={`${rowIndex}-${column.key}`}
										className={`px-6 py-3 whitespace-nowrap  text-gray-400 ${column.className || ""}`}>
										{column.render ? column.render(row[column.key], row) : row[column.key]}
									</td>
								))}
							</tr>
						))}
					</tbody>
				</table>
			</div>

			{/* Mobile Card View */}
			<div className="md:hidden space-y-4">
				{data.map((row, rowIndex) => (
					<div key={rowIndex} className="bg-sidebar-bg border border-gray-700 rounded-lg p-4 hover:bg-gray-700">
						{columns.map((column) => (
							<div
								key={`${rowIndex}-${column.key}`}
								className="flex justify-between items-start py-2 border-b border-gray-700 last:border-0">
								<div className="text-xs font-medium text-gray-400 uppercase">{column.header}</div>
								<div className="text-sm text-right text-white ml-4">
									{column.render ? column.render(row[column.key], row) : row[column.key]}
								</div>
							</div>
						))}
					</div>
				))}
			</div>
		</div>
	);
};
