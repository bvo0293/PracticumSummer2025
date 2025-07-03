import type { CollabRecProps } from "./types";

function Plot({ collab_rec }: CollabRecProps) {
  const rows = collab_rec ? collab_rec : [];
  if (Object.keys(rows).length === 0) return <p></p>;

return (
  <div>
      <h3>🤝 Courses Popular Among Academic Peers</h3>
      <p>
        Based on courses that students with a similar academic profile to yours have succeeded in.
      </p>

  <ul>
    {rows.map((item) => (
      <li key={item.CourseNumber}>
        <strong> {item.coursename} :</strong> {item.HonorsDesc}  (ID: {item.CourseNumber}) — Taken by {item.peer_count} peers.
      </li>
    ))}
  </ul>
  </div>

);
}

export default Plot;
