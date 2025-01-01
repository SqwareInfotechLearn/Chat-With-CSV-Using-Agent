import os
from dotenv import load_dotenv
from pathlib import Path
from phi.agent.python import PythonAgent
from phi.file.local.csv import CsvFile
from phi.model.groq import Groq
import streamlit as st

# Load environment variables for API keys and configurations
load_dotenv()

# Set up the base directory for temporary files
cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
if not tmp.exists():
    tmp.mkdir(exist_ok=True, parents=True)

# Path to the local IMDB CSV file
local_csv_path = "IMDB-Movie-Data.csv"

# Ensure the CSV file exists in the project directory
if not os.path.exists(local_csv_path):
    raise FileNotFoundError(f"The file {local_csv_path} does not exist. Please ensure it is placed in the project directory.")

# Configure the PythonAgent with Groq's Llama model and the local CSV file
python_agent = PythonAgent(
    model=Groq(id="llama-3.1-70b-versatile"),  # Use Groq's Llama 3.1 70B Versatile model
    base_dir=tmp,  # Temporary directory for intermediate files
    files=[
        CsvFile(
            path=local_csv_path,  # Integrate the IMDB CSV file
            description="A dataset containing information about movies, genres, directors, and IMDB ratings.",
        )
    ],
    markdown=True,  # Enable markdown formatting for responses
    pip_install=True,  # Install required dependencies automatically
    show_tool_calls=True,  # Display tool calls for better transparency
)

# Main function for the Streamlit app
def main():
    st.title("CSV Agent - Chat with CSV data")  # App title
    
    st.write("Welcome! Ask questions about the IMDB movie dataset, and I’ll fetch the answers for you.")  # App description
    
    question = st.text_area("Enter your question:", placeholder="e.g., Which movie has the highest IMDB rating?")
    
    if st.button("Run Flow"):
        if not question.strip():
            st.error("Please enter a valid question.")
            return
        
        try:
            with st.spinner("Processing your question..."):  # Show a loading spinner
                response = python_agent.run(question)  # Run the agent with the user query
                st.markdown(response.content)  # Display the response in markdown format
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")  # Handle and display any errors

if __name__ == "__main__":
    main()
