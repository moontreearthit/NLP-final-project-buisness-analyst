import pandas as pd
import json
import google.generativeai as genai
import streamlit as st
import plotly.express as px

st.title("Customer Review AI Analyzer")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
file = st.sidebar.file_uploader("Upload CSV (must contain 'review_text' column). Compatable with TH/EN",type="csv")

def analyze_review(review_text: str, api_key: str):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    You are an expert Customer Review Analyst. Your task is to analyze customer feedback and extract structured business intelligence.
    pain point summary and business recomendation should show in Thai language
    Return ONLY valid JSON. No ``` no explanation.

{{
  "sentiment_score": 1-5,
  "urgency_score": 1-5,
  "category": "สุขลักษณะ | บริการ | ความรวดเร็ว | ราคา | สถานที่และบรรยากาศ | อื่น ๆ",
  "pain_point_summary": "...",
  "business_recommendation": "..."
}}

Review:
{review_text}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        text = text[text.find("{"): text.rfind("}") + 1]
        return json.loads(text)
    except Exception as e:
        return {
            "sentiment_score": 0,
            "urgency_score" : 0,
            "category": "Error",
            "pain_point_summary": "LLM error",
            "business_recommendation": "Check API key"
        }
    
def process_reviews(df, api_key):
    results =  []
    progress = st.progress(0)

    for i, row in df.iterrows():
        analysis = analyze_review(row["review_text"], api_key)

        results.append({
            "review_num": row.get("review_id", i),
            "Review_text": row["review_text"],
            "Sentiment_Score": analysis["sentiment_score"],
            "Urgency_Score": analysis["urgency_score"],
            "Issue_Category": analysis["category"],
            "Pain_Point": analysis["pain_point_summary"],
            "Recommendation": analysis["business_recommendation"]
        })

        progress.progress((i+1)/len(df))
    return pd.DataFrame(results)


if file and api_key:
    df_raw = pd.read_csv(file)
    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head())

    if st.button("🚀 Start Analysis"):
        if "review_text" not in df_raw.columns:
            st.error("your csv file must contain 'review_text' column")
        else:
            df_result = process_reviews(df_raw, api_key)

            st.success("Complete!")

            # ------------------------- Visualization -------------------------
            st.header("📈 Sentiment vs Urgency")
            fig = px.scatter(
                df_result,
                x="Sentiment_Score",
                y="Urgency_Score",
                size="Urgency_Score",
                color="Issue_Category",
                hover_name="Pain_Point",
                title="Issue Mapping"
            )
            fig.update_xaxes(range=[0.5, 5.5], dtick=1)
            fig.update_yaxes(range=[0.5, 5.5], dtick=1)
            st.plotly_chart(fig, use_container_width=True)

            # ------------------------- Summary Table -------------------------
            st.header("📝 Summary")
            summary = df_result.groupby("Issue_Category").agg({
                "review_id": "count",
                "Sentiment_Score": "mean",
                "Urgency_Score": "mean"
            }).round(2)
            st.dataframe(summary)

            # ------------------------- Export CSV -------------------------
            st.header("📄 Analyzed Data")
            st.dataframe(df_result)

            csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                csv,
                "analyzed_reviews.csv",
                "text/csv"
            )

elif file and not api_key:
    st.warning("please enter API key")
    