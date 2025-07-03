import type { MLRecProps } from "./types";

function Plot({ ml_rec, needed_credits }: MLRecProps) {
  console.log(ml_rec);
  console.log(needed_credits);

  if (Object.keys(ml_rec).length === 0) return <p></p>;

  return (
    <div>
      <h3>🤖 Graduation Credit & AI-Powered Course Recommendations</h3>
      <p>
        Recommendations to fill credit gaps, with a predicted success score.
      </p>

    <ul>
      {needed_credits.map((credit) => {
        const recommendations = ml_rec.filter(
          (rec) => rec.SubjectArea === credit.SubjectArea
        );

        const needsCredits = credit.AreaCreditStillNeeded > 0;

        return (
          <li key={credit.SubjectArea} className="mb-2">
            <strong>{credit.SubjectArea}:</strong>{" "}
            {needsCredits ? (
              <>
                ❗ needs {credit.AreaCreditStillNeeded} credits —{" "}
                {recommendations.length > 0 ? (
                  <>
                    here are the top recommended courses:
                    <ul className="pl-4 mt-1">
                      {recommendations.map((rec) => (
                        <li key={rec.CourseId}>
                          {rec.coursename} ({rec.HonorsDesc}) — success probability:{" "}
                          {(rec.success_prob * 100).toFixed(1)}%
                        </li>
                      ))}
                    </ul>
                  </>
                ) : (
                  <>No specific course recommendations generated for this subject.</>
                )}
              </>
            ) : (
              <>✅ complete</>
            )}
          </li>
        );
      })}
    </ul>
    </div>
  );
}

export default Plot;
