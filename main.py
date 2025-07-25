import streamlit as st
import pandas as pd
import anthropic
import os

def load_data():
   """Load data with different encodings"""
   encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
   
   for encoding in encodings_to_try:
       try:
           df = pd.read_csv('Cleaned_Matters_OGC.csv', encoding=encoding)
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
       # Skip if essential information is NA or empty
       if (pd.isna(lawyer['Attorney']) or 
           pd.isna(lawyer['Work Email']) or 
           pd.isna(lawyer['Education']) or 
           pd.isna(lawyer['Summary and Expertise']) or
           lawyer['Attorney'].strip().lower() == 'na' or
           lawyer['Work Email'].strip().lower() == 'na' or
           lawyer['Education'].strip().lower() == 'na' or
           lawyer['Summary and Expertise'].strip().lower() == 'na'):
           continue
           
       with cols[idx % 3]:
           with st.expander(f"🧑‍⚖️ {lawyer['Attorney']}", expanded=False):
               expertise_bullets = [f"• {area.strip()}" for area in str(lawyer['Summary and Expertise']).split(',')]
               expertise_text = "\n".join(expertise_bullets)
               
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
   """Get Claude's analysis of the best lawyer matches with domain knowledge"""
   try:
       # Initialize client with Streamlit secret
       client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
       
       summary_text = "Available Lawyers and Their Expertise:\n\n"
       for _, lawyer in lawyers_df.iterrows():
           summary_text += f"- {lawyer['Attorney']}\n"
           summary_text += f"  Education: {str(lawyer['Education'])}\n"
           summary_text += f"  Expertise: {str(lawyer['Summary and Expertise'])}\n\n"

       system_prompt = "You are a legal staffing specialist helping to match lawyers with client needs."
       
       message = client.messages.create(
           model="claude-3-5-sonnet-20241022",  # Updated to Claude 3.5 Sonnet (latest stable)
           system=system_prompt,
           temperature=0.1,
           max_tokens=1500,
           messages=[{
               "role": "user", 
               "content": f"""Client Need: {query}

{summary_text}

Please analyze the lawyers' profiles and recommend matches based on these important guidelines:

1. For employment law queries:
  - Patricia Lantzy, Margaret Scheele, Sarah Biran, and Lorna Hebert are the key employment attorneys
  - Note that no employment lawyer works more than 85 hours per month

2. For corporate formation:
  - Key experts include Kristin Kreuder, Michael Mendelson, Bruce Friedman, Leonard McGill, Nicole Desharnais, Chandana Rao, Caroline McCaffrey
  - Susan Antonio has expertise but limited availability

3. For trademark work:
  - Primary experts are Wade Savoy, Michelle Roseberg, and Jessica Davis

4. For HIPAA and BAAs:
  - Core experts are Holly Little, Bob Michitarian, Mark Feingold, Michael Brown, Marty Lipman
  - Fritz Backus and Raymond Sczudlo have limited HIPAA experience
  - Brian Heller should be noted as having NO HIPAA experience

5. For specialized areas:
  - FDA/Pharma: Mark Mansour (primary), Berry Cappucci, Jordan Karp, Holly Little, Elizabeth Smith (limited)
  - Data Privacy: Caroline McCaffrey, Mark Johnson, Lori Ross, Stephan Grynwajc, Lakshmi Ramani
  - Real Estate: Josh Miller, James Duberman, Michael Plantamura
  - Social Media/Influencer: Stacey Heller, Chandana Rao, Joseph Tedeschi, Bruce Friedman, Ted Stern, Brad Auerbach
  - Satellite/Aerospace: Michael Mendelson, Don Levy, Donnellda Rice, Ron Jarvis
  - DAFs: Anita Drummond, Lakshmi Ramani
  - Content/Media Licensing: Ted Stern, Brian Heller, Joseph Tedeschi, Chandana Rao, Andy Friedman
  - Commercial Real Estate: Josh Miller, James Duberman, Michael Plantamura
  - Equity Compensation: Nicole Desharnais, Leonard McGill, Don Levy
  - Retail Industry: Stacey Heller (big box), Billie Audia Munro (big box), Chandana Rao (fashion)

Please provide matches in this exact format:

MATCH_START
Rank: 1
Name: [Attorney Name]
Key Expertise: [Relevant expertise for this query]
Recommendation Reason: [Why this lawyer is appropriate, including any caveats or limitations]
MATCH_END"""
           }]
       )
       
       # Extract the content from the first message
       response_text = message.content[0].text
       return parse_claude_response(response_text)

   except Exception as e:
       st.error("Error getting recommendations")
       st.sidebar.error(f"API Error Details: {str(e)}")
       st.sidebar.error(f"Error Type: {type(e)}")
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
       desired_columns = ['Rank', 'Name', 'Key Expertise', 'Recommendation Reason']
       existing_columns = [col for col in desired_columns if col in df.columns]
       df = df[existing_columns]
       if 'Rank' in df.columns:
           df['Rank'] = pd.to_numeric(df['Rank'])
           df = df.sort_values('Rank')
   
   return df

def main():
   st.title("🧑‍⚖️ Outside GC Lawyer Matcher")
   
   try:
       lawyers_df = load_data()
       if lawyers_df is None:
           st.error("Failed to load lawyer data.")
           return
           
       # Debug info
       if st.sidebar.checkbox("Show Data Info"):
           st.sidebar.write("Data Shape:", lawyers_df.shape)
           st.sidebar.write("Columns:", list(lawyers_df.columns))
           st.sidebar.write("Sample Data:", lawyers_df.head())
       
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
       
       st.write("### How can we help you find the right lawyer?")
       
       # Example queries based on common requests
       examples = [
           "Show me an available employment attorney",
           "Who can help with HIPAA and BAAs?",
           "I need a lawyer for trademark work",
           "Looking for someone with FDA and pharmaceutical experience",
           "Need help with data privacy and cybersecurity",
           "Who can help with social media and influencer agreements?",
           "Looking for a commercial real estate lawyer",
           "Need help with equity compensation plans"
       ]
       
       # Example query buttons
       col1, col2 = st.columns(2)
       for i, example in enumerate(examples):
           if i % 2 == 0:
               if col1.button(f"🔍 {example}"):
                   st.session_state.query = example
                   st.rerun()  # Updated from st.experimental_rerun()
           else:
               if col2.button(f"🔍 {example}"):
                   st.session_state.query = example
                   st.rerun()  # Updated from st.experimental_rerun()

       # Filter lawyers based on selection
       filtered_df = lawyers_df.copy()
       if selected_practice_area != "All":
           filtered_df = filtered_df[
               filtered_df['Summary and Expertise'].str.contains(str(selected_practice_area), na=False, case=False)
           ]
       
       # Custom query input
       query = st.text_area(
           "Describe what you're looking for:",
           value=st.session_state.get('query', ''),
           placeholder="Example: I need an employment lawyer with retail industry experience...",
           height=100
       )

       # Search and Clear buttons
       col1, col2 = st.columns([1, 4])
       search = col1.button("🔎 Search")
       clear = col2.button("Clear")

       if clear:
           st.session_state.query = ''
           st.rerun()  # Updated from st.experimental_rerun()

       # Show counts
       st.sidebar.markdown("---")
       st.sidebar.markdown(f"**Showing:** {len(filtered_df)} lawyers")
       
       # Show recommendations or all lawyers
       if search and query:
           with st.spinner("Finding the best matches..."):
               results_df = get_claude_response(query, filtered_df)
               if results_df is not None and not results_df.empty:
                   st.markdown("### 🎯 Top Lawyer Matches")
                   # Updated dataframe display with custom height
                   st.dataframe(
                       results_df,
                       hide_index=True,
                       use_container_width=True,
                       height=400  # Increased height in pixels
                   )
                   
                   # Add disclaimer
                   st.info("""
                   ℹ️ These recommendations are based on known expertise and availability. 
                   Please confirm specific details and availability with the lawyers directly.
                   """)
       else:
           create_lawyer_cards(filtered_df)
           
   except FileNotFoundError:
       st.error("Could not find the required data file. Please check the file location.")
   except Exception as e:
       st.error(f"An error occurred: {str(e)}")
       if st.sidebar.checkbox("Show Debug Info"):
           st.sidebar.error(f"Error details: {str(e)}")

if __name__ == "__main__":
   main()
