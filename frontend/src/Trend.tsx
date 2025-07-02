import type { TrendProps } from "./types";

function Plot({ trend }: TrendProps) {
  const rows = trend[0] ? trend[0] : [];
  if (Object.keys(rows).length === 0) return <p></p>;


  return (
    <ul >
      {Object.entries(rows).map(([department, description]) => (
        <li key={department}>
          <strong>{department}:</strong> {description}
        </li>
      ))}
    </ul>
  );
}

export default Plot;
