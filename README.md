# Langsmith Project Setup Guide

## 1. Create Project Folder

-   Create a project folder named **Langsmith**
-   Locate or clone the project code from GitHub into this folder

------------------------------------------------------------------------

## 2. Create and Activate Virtual Environment

### a. Create virtual environment

``` bash
python -m venv labenv
```

### b. Activate virtual environment (Windows)

``` bash
.\labenv\Scripts\Activate
```

------------------------------------------------------------------------

## 3. Install Requirements

Run the following command:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 4. Create OpenAI API Key

-   Visit: https://platform.openai.com/settings/organization/api-keys
-   Generate a new API key
-   Save it securely

------------------------------------------------------------------------

## 5. Create LangSmith API Key

-   Visit: https://smith.langchain.com/
-   Sign in and generate your LangSmith API key
-   Save it securely

------------------------------------------------------------------------

# Deployment using FAST API

## 1. Update requirements.txt

Add the following:

``` txt
fastapi
uvicorn[standard]
```

------------------------------------------------------------------------

## 2. Install Updated Requirements

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 3. Run the Application

``` bash
python -m uvicorn < pythonfilename >:app --reload
```

``` bash
python -m uvicorn app_llm:app --reload
```

------------------------------------------------------------------------

## 4. Open in Browser to Test

Open: http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## 5. Test with JSON

Use the following JSON body:

``` json
{
  "question": "What is AI?"
}
```
