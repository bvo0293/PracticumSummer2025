import type { CardProps } from "./types";

function Cards({ data }: CardProps) {
  const rows = data[0] ? data[0] : [];
  if (Object.keys(rows).length === 0) return <p></p>;

  return (
    <div>
      <h3>🎓 Total Credits Earned by Department</h3>
      <p>
        This session shows the total credits students have earned, broken down
        by academic department.
      </p>
      <div className="container mt-4">
        <div className="row justify-content-start">
          {Object.entries(rows).map(([subject, credit], idx) => (
            <div className="card-col mb-3" key={idx}>
              <div className="card h-100 border-0">
                <div className="card-body">
                  <h5 className="card-title" style={{fontSize: "12px", color:"gray", height:"25px"}}>{subject}</h5>
                  <p className="card-text" style={{fontSize: "30px", color:"black"}}>{credit.toFixed(2)} credits</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default Cards;
