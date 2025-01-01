# Chat with CSV Using Agentic AI

This project leverages **Agentic AI**, **Phidata**, **Groq**, and **Streamlit** to enable an interactive, chat-based interface for querying data from a CSV file. Specifically, it uses the IMDB Movie dataset (`IMDB-Movie-Data.csv`) and allows users to ask natural language questions about the dataset. The solution integrates the powerful Groq model to process questions efficiently, making it possible to extract insights from large CSV files with ease.

### Key Features:
- **Chat-based Interaction**: You can ask questions about the IMDB movie dataset in natural language, and the model will process your query and return answers.
- **CSV Data Integration**: It uses the IMDB movie dataset in CSV format, including movie names, genres, directors, ratings, and other details.
- **Powered by Agentic AI**: Uses **PhiAgent** with **Groq's Llama model** to provide intelligent, context-aware answers based on the CSV data.
- **Streamlit Interface**: A user-friendly web interface that allows you to interact with the dataset easily and ask questions without writing code.

### Technologies:
- **Agentic AI**: Enables querying and interaction with the CSV data using natural language.
- **Groq**: High-performance hardware and software stack for machine learning, used here to accelerate model inference and processing.
- **Phidata**: Data management framework that facilitates the integration of CSV files.
- **Streamlit**: Creates a simple, interactive user interface to chat with the data.

### Project Setup:

#### Prerequisites:
- Python 3.7+
- A valid `.env` file with any required API keys.
- The **IMDB-Movie-Data.csv** file should be in the root directory of the project (or update the path in the code accordingly).

#### 1. Clone the Repository:
```bash
git clone https://github.com/SqwareInfotechLearn/Chat-With-CSV-Using-Agent.git
cd Chat-With-CSV-Using-Agent
```

#### 2. Set Up the Environment:
Create a `.env` file in the root directory to store any necessary environment variables such as API keys (e.g., for Groq or any other service).

Example `.env` file (adjust based on your actual API requirements):
```env
GROQ_API_KEY=your_api_key_here
```

#### 3. Download the Dataset:
Ensure the **IMDB-Movie-Data.csv** file is placed in the project directory or modify the `local_csv_path` variable in the code to point to the correct file path.

#### 4. Run the Streamlit App:
Start the app with Streamlit:
```bash
streamlit run app.py
```

This will launch a local development server and you can interact with the app in your browser.

---

### How It Works:

1. **Streamlit Interface**: 
   - The user enters a question in the provided text area (e.g., "Which movie has the highest IMDB rating?").
   - The user presses the "Run Flow" button to send the query.

2. **Agentic AI**: 
   - The Python agent uses **Groq's Llama model** to process the user's question, integrating natural language understanding to extract insights from the IMDB CSV dataset.
   
3. **Groq Model**: 
   - Groq's hardware-accelerated Llama model is used to efficiently process queries and return results quickly, even with large datasets.

4. **Response**:
   - The agent's response is displayed in the Streamlit interface in markdown format.

---

### Example Questions You Can Ask:
- "Which movie has the highest IMDB rating?"
- "What are the genres of the movie 'Inception'?"
- "Who directed the movie 'The Dark Knight'?"
- "What is the average IMDB rating of all movies?"
- "How many movies were released in 2000?"

---

### Acknowledgements:
- **Groq**: Hardware accelerators used for fast inference.
- **Phidata**: Data integration framework.
- **Streamlit**: For creating the user-friendly web interface.
- **Agentic AI**: Enabling natural language interactions with the data.
