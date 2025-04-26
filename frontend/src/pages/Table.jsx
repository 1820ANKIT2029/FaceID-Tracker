import React from "react";

function Table({ data }) {
  console.log(data)
  return (
    <div className="overflow-x-auto mt-6 bg-white/5 backdrop-blur-md border border-cyan-500 rounded-xl p-4 shadow-md">
      <h3 className="text-xl font-semibold mb-4 text-cyan-300">Detection Results</h3>
      <table className="min-w-full text-white text-center">
        <thead>
          <tr className="bg-cyan-600/20 border-b border-cyan-400">
            <th className="py-3 px-6">Name</th>
            <th className="py-3 px-6">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {data.length === 0 ? (
            <tr>
              <td colSpan="2" className="py-6 text-cyan-200">
                No results yet.
              </td>
            </tr>
          ) : (
            data.map((item, index) => (
              <tr
                key={index}
                className="hover:bg-cyan-500/10 transition duration-200 border-t border-cyan-800"
              >
                <td className="py-2 px-6 font-medium">{item.name}</td>
                <td className="py-2 px-6">{item?.confidence?.toFixed(2)}%</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export default Table;