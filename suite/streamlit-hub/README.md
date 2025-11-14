# MLAI Streamlit Hub

This folder contains a central Streamlit app that serves as a hub for all
Machine Learning & AI Engineering hero apps in your portfolio.

## What it does

- Lists each hero app with a title and description.
- Shows the local `streamlit run ...` command to launch each app from the
  repository root.
- Allows you to optionally add GitHub and Hostinger URLs for each app so
  recruiters can click through.

## Run Locally

From the root of your `mlai-portfolio` repo:

```bash
pip install -r suite/streamlit-hub/requirements.txt
streamlit run suite/streamlit-hub/app.py
```

Then open the URL shown in the terminal.
