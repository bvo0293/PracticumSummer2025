import streamlit as st
import streamlit.components.v1 as components

mermaid_code = """
graph TD
    subgraph "Offline Data Preparation"
        A1[Raw CSVs: Final_DF.csv, Courses.csv, GraduationAreaSummary.csv, Illuminate2025.csv] --> B(Data Prep Scripts: EDA.ipynb, prepare_illuminate_data.py);
        B --> C{Clean, Filter, Merge, and Pivot Data};
        C --> D1[Cleaned Files: cleaned_illuminate.csv, training_data_v2.csv];
        C --> D2[Trained ML Model: student_success_model.pkl];
    end

    subgraph "Live Streamlit Application (streamlit_student_dashboard_v5_final.py)"
        E[User Interface: Streamlit Sidebar] --> F(User inputs Student ID);
        F --> G{View Performance Button Clicked};

        subgraph "On-Demand Processing"
            G --> H1[ML Recommendation Engine];
            G --> H2[Collaborative Filtering Engine];
            G --> H3[Performance Analysis Engine];
        end

        subgraph "Cached Assets (Loaded once)"
            D1 --> I(Load Pre-processed Data);
            D2 --> I;
        end

        I --> H1;
        I --> H2;
        I --> H3;

        H1 --> J1[Graduation & AI Recs];
        H2 --> J2[Peer Recommendations];
        H3 --> J3[Performance Analysis & Charts];
        I -- Illuminate Data --> J4[Current Assessment Snapshot];

        J1 --> K[Streamlit UI Display];
        J2 --> K;
        J3 --> K;
        J4 --> K;
    end

    style B fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#ccf,stroke:#333,stroke-width:2px
    style I fill:#fcf,stroke:#333,stroke-width:2px
    style K fill:#9f9,stroke:#333,stroke-width:2px
"""

components.html(f"""
<!DOCTYPE html>
<html>
<head>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true }});
  </script>
</head>
<body>
  <div class="mermaid">
    {mermaid_code}
  </div>
</body>
</html>
""", height=1000, scrolling=True)