import React from "react";

const Table = ({data}) => {

  return (
    <div className="mt-8">
      <h2 className="text-xl font-semibold mb-4">Attendance List</h2>
      {data.length === 0 ? (
        <p className="text-gray-500">No attendance recorded yet.</p>
      ) : (
        <ul className="divide-y divide-gray-200">
          {data.map((item, index) => (
            <li key={index} className="py-2">
              {item.name} - {item.time}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default Table;