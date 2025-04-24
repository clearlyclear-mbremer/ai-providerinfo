# ClearlyClear AI Bot

This is a Flask-based AI assistant that uses LangChain and OpenAI to answer questions using content pulled from your Confluence space.

## Setup

1. Add environment variables:
   - OPENAI_API_KEY
   - CONFLUENCE_URL (e.g., https://yourcompany.atlassian.net/wiki)
   - CONFLUENCE_USERNAME
   - CONFLUENCE_API_KEY

2. Run `embed_docs.py` to embed your Confluence pages.

3. Deploy to Render or Railway using:
   - `gunicorn app.app:app` as your start command
   - Add persistent disk for ./chroma_store if needed

4. Use the /ask endpoint to send POST requests with a `question`.

## Example

```bash
curl -X POST http://your-app.onrender.com/ask -H "Content-Type: application/json" -d '{"question":"How do I install the Invisalign software?"}'
```
