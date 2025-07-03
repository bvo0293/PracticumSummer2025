import type { PlotImageProps } from "./types";

function Plot({ image }: PlotImageProps) {
  if (!image) return <p></p>;

  return (
    <div>
      <h3>📈 Average Mark Over Time</h3>
      <p>
        This session visualizes how a student’s average mark changes over time across different grading periods.
      </p>
      <div className="d-flex justify-content-between align-items-center">
        <img
          src={`data:image/png;base64,${image}`}
          style={{ maxWidth: "100%" }}
        />
      </div>
    </div>
  );
}

export default Plot;
