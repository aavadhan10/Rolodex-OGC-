import streamlit as st
import pandas as pd
from anthropic import Anthropic
import os

# Initialize Anthropic with API key
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def load_data():
    """Load data with different encodings"""
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv('Cleaned_Matters_OGC.csv', encoding=encoding)
            # Only select the columns we want
            return df[['Attorney', 'Work Email', 'Education', 'Summary and Expertise']]
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"Error reading file with {encoding} encoding: {str(e)}")
            continue
    
    raise Exception("Could not read the CSV file with any of the attempted encodings")

def create_lawyer_cards(lawyers_df):
    """Create card layout for lawyers"""
    if lawyers_df.empty:
        st.warning("No lawyers match the selected filters.")
        return
        
    st.write("### 📊 Available Lawyers")
    
    lawyers_df = lawyers_df.sort_values('Attorney')
    cols = st.columns(3)
    
    for idx, (_, lawyer) in enumerate(lawyers_df.iterrows()):
        with cols[idx % 3]:
            with st.expander(f"🧑‍⚖️ {lawyer['Attorney']}", expanded=False):
                # Format expertise with bullet points
                expertise_bullets = [f"• {area.strip()}" for area in str(lawyer['Summary and Expertise']).split(',')]
                expertise_text = "\n".join(expertise_bullets)
                
                # Create markdown content
                content = f"""
**Contact:**
{lawyer['Work Email']}

**Education:**
{lawyer['Education']}

**Expertise:**
{expertise_text}
"""
                st.markdown(content)

def get_claude_response(query, lawyers_df):
    """Get Claude's analysis of the best lawyer matches"""
    summary_text = "Available Lawyers and Their Expertise:\n\n"
    for _, lawyer in lawyers_df.iterrows():
        summary_text += f"- {lawyer['Attorney']}\n"
        summary_text += f"  Education: {str(lawyer['Education'])}\n"
        summary_text += f"  Expertise: {str(lawyer['Summary and Expertise'])}\n\n"

    prompt = f"You are a legal staffing assistant. Your task is to match client needs with available lawyers based on their expertise and background.\n\nClient Need: {query}\n\n{summary_text}\nPlease analyze the lawyers' profiles and provide the best 3-5 matches in this format:\n\nMATCH_START\nRank: 1\nName: John Smith\nKey Expertise: Corporate Law, M&A\nEducation: Harvard Law School J.D.\nRecommendation Reason: Extensive experience in corporate transactions\nMATCH_END"

    try:
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return parse_claude_response(response.content[0].text)
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None

def parse_claude_response(response):
    """Parse Claude's response into a structured format"""
    matches = []
    for match in response.split('MATCH_START')[1:]:
        match_data = {}
        lines = match.split('MATCH_END')[0].strip().split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                match_data[key.strip()] = value.strip()
        
        if match_data:
            matches.append(match_data)
    
    df = pd.DataFrame(matches)
    if not df.empty:
        desired_columns = ['Rank', 'Name', 'Key Expertise', 'Education', 'Recommendation Reason']
        existing_columns = [col for col in desired_columns if col in df.columns]
        df = df[existing_columns]
        if 'Rank' in df.columns:
            df['Rank'] = pd.to_numeric(df['Rank'])
            df = df.sort_values('Rank')
    
    return df

def main():
    st.title("🧑‍⚖️ Outside GC Lawyer Matcher")
    
    try:
        # Load data with encoding handling
        lawyers_df = load_data()
        if lawyers_df is None:
            st.error("Failed to load lawyer data.")
            return
            
        # Debug info
        if st.sidebar.checkbox("Show Data Info"):
            st.sidebar.write("Data Shape:", lawyers_df.shape)
            st.sidebar.write("Columns:", list(lawyers_df.columns))
            st.sidebar.write("Sample Data:", lawyers_df.head())
        
        # Sidebar filters
        st.sidebar.title("Filters")
        
        # Get practice areas
        practice_areas = []
        for expertise in lawyers_df['Summary and Expertise'].dropna():
            practice_areas.extend([area.strip() for area in str(expertise).split(',')])
        practice_areas = sorted(list(set(practice_areas)))
        
        # Practice area filter
        selected_practice_area = st.sidebar.selectbox(
            "Practice Area",
            ["All"] + practice_areas
        )
        
        # Main content area
        st.write("### How can we help you find the right lawyer?")
        st.write("Tell us about your legal needs and we'll match you with the best available lawyers.")
        
        # Example queries
        examples = [
            "I need a lawyer experienced in intellectual property and software licensing",
            "Looking for someone who handles business startups and corporate governance",
            "Need help with technology transactions and SaaS agreements",
            "Who would be best for mergers and acquisitions in the technology sector?"
        ]
        
        # Example query buttons
        col1, col2 = st.columns(2)
        for i, example in enumerate(examples):
            if i % 2 == 0:
                if col1.button(f"🔍 {example}"):
                    st.session_state.query = example
                    st.rerun()
            else:
                if col2.button(f"🔍 {example}"):
                    st.session_state.query = example
                    st.rerun()

        # Filter lawyers based on selection
        filtered_df = lawyers_df.copy()
        if selected_practice_area != "All":
            filtered_df = filtered_df[
                filtered_df['Summary and Expertise'].str.contains(str(selected_practice_area), na=False, case=False)
            ]
        
        # Custom query input
        query = st.text_area(
            "For more specific matching, describe what you're looking for:",
            value=st.session_state.get('query', ''),
            placeholder="Example: I need help with intellectual property and software licensing...",
            height=100
        )

        # Search and Clear buttons
        col1, col2 = st.columns([1, 4])
        search = col1.button("🔎 Search")
        clear = col2.button("Clear")

        if clear:
            st.session_state.query = ''
            st.rerun()

        # Show counts
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Showing:** {len(filtered_df)} lawyers")
        
        # Show recommendations or all lawyers
        if search and query:
            with st.spinner("Finding the best matches..."):
                results_df = get_claude_response(query, filtered_df)
                if results_df is not None and not results_df.empty:
                    st.markdown("### 🎯 Top Lawyer Matches")
                    st.dataframe(
                        results_df,
                        hide_index=True,
                        use_container_width=True
                    )
        else:
            create_lawyer_cards(filtered_df)
            
    except FileNotFoundError:
        st.error("Could not find the required data file. Please check the file location.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if st.sidebar.checkbox("Show Debug Info"):
            st.sidebar.error(f"Error details: {str(e)}")
            st.sidebar.write("Stack trace:", e.__traceback__)

if __name__ == "__main__":
    main()
