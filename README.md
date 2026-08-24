# Blog_Writing_Agent

An **Agentic AI-powered Blog Writing Agent** that automatically researches, plans, writes, and formats technical blog posts using a multi-agent workflow built with **LangGraph**.

The application provides a **Streamlit interface** where users can enter a topic, optionally enable web research and AI image generation, and generate a complete Markdown blog.

## 🎥 Demo

A demo video of the application is included in the project.

## ✨ Features

- 🧠 **Agentic AI workflow** using LangGraph
- 🧭 **Intelligent topic routing**
- 🔎 **Optional web research** using Tavily
- 📝 **Multi-agent blog planning and writing**
- ⚡ **Parallel section generation**
- 🔗 **Evidence-based citations** for research-driven content
- 🖼️ **Optional AI image generation**
- 📄 **Markdown blog generation**
- 🖥️ **Streamlit web interface**
- 💰 **Token-aware architecture** optimized for limited API usage

## 🏗️ Architecture

```text
                    User
                     │
                     ▼
              ┌──────────────┐
              │   Streamlit  │
              │   Frontend   │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │    Router    │
              └──────┬───────┘
                     │
             ┌───────┴────────┐
             │                │
             ▼                ▼
        Closed Book       Web Research
             │                │
             │             Tavily
             │                │
             └───────┬────────┘
                     ▼
              ┌──────────────┐
              │ Orchestrator │
              └──────┬───────┘
                     │
             ┌───────┼───────┐
             ▼       ▼       ▼
          Worker   Worker   Worker
             │       │       │
             └───────┼───────┘
                     ▼
              ┌──────────────┐
              │   Reducer    │
              └──────┬───────┘
                     │
               Generate Images?
                  /       \
                No         Yes
                │           │
                │         Gemini
                │           │
                └─────┬─────┘
                      ▼
                Final Markdown
````

## 🔄 Workflow

### 1. User Input

The user enters a blog topic through the Streamlit interface.

Example:

```text
What is Retrieval Augmented Generation (RAG)?
```

The user can also choose whether to generate AI images.

### 2. Router

The router determines whether external research is required.

It supports three modes:

* `closed_book` — uses the model's existing knowledge
* `hybrid` — combines model knowledge with limited web research
* `open_book` — performs web research for current or time-sensitive topics

### 3. Research

When research is required, **Tavily** retrieves relevant sources.

The system intentionally limits the number of searches and results to reduce token and API usage.

### 4. Orchestrator

The orchestrator creates a structured blog plan and divides the article into multiple sections.

### 5. Worker Agents

Multiple worker agents independently generate the planned sections.

Each worker receives:

* Section objective
* Key points
* Target word count
* Research evidence when required
* Citation requirements
* Code requirements

### 6. Reducer

The reducer combines the generated sections into a single coherent Markdown article.

### 7. Optional Image Generation

Users can enable image generation from the frontend.

When enabled:

```text
Blog
 ↓
Image Planner
 ↓
Google Gemini
 ↓
Generated Image
 ↓
Final Markdown
```

When disabled, the image-generation stage is skipped completely.

## 🛠️ Tech Stack

* **Python**
* **LangGraph**
* **LangChain**
* **Groq**
* **Tavily**
* **Google Gemini**
* **Pydantic**
* **Streamlit**

## 📁 Project Structure

```text
Blog_Writing_Agent/
│
├── BWA_backend.py          # LangGraph agentic backend
├── BWA_frontend.py         # Streamlit frontend
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignored files
│
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vijay7300/Blog_Writing_Agent.git
cd Blog_Writing_Agent
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv myenv
```

Activate it:

```powershell
myenv\Scripts\activate
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_API_KEY=your_google_api_key
```

> Never commit your `.env` file or expose API keys publicly.

## 🚀 Run the Application

Start the Streamlit frontend:

```powershell
python -m streamlit run BWA_frontend.py
```

Then open:

```text
http://localhost:8501
```

## 🖥️ Example

Enter a topic such as:

```text
What is Retrieval Augmented Generation (RAG)?
```

Then choose:

```text
🖼️ Generate Images
```

if visual content is required.

The system generates a structured technical blog and provides the final Markdown output through the Streamlit interface.

## 🎯 Key Design Decisions

### Conditional Research

The system does not perform web searches for every topic. This reduces unnecessary API usage.

### Parallel Writing

Instead of generating the entire blog in a single LLM call, the article is divided into sections and generated by multiple worker nodes.

### Token Optimization

The system limits:

* Number of research queries
* Number of retrieved sources
* Evidence snippet length
* Number of generated tasks
* Image-planning complexity

This helps the application operate within limited LLM API quotas.

### Optional Image Generation

Image generation is user-controlled, preventing unnecessary image API usage when visuals are not required.

## 📌 Future Improvements

* Persistent blog history
* More advanced citation verification
* Additional LLM providers
* Streaming token-level generation
* PDF/HTML export
* Improved image generation pipeline
* Deployment to a cloud platform
* User authentication
* Blog style and audience customization

## 👨‍💻 Author

**Vijay Prajapati**
M.Sc-M.Tech at IIT Jodhpur (published on 25-8-26)

GitHub:
[https://github.com/Vijay7300](https://github.com/Vijay7300)

## 📄 License

This project is available under the MIT License.

````

### One correction I'd make before committing

Because your repository already has a `README.md` and `LICENSE`, **don't create another README**. Replace the existing `README.md` with the content above, then:

```powershell
git add README.md
git commit -m "Improve project README"
git push
````

Then your GitHub repository will have a much more professional presentation.
