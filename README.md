# PlantCareAI

Production-ready split for:

- `plant-disease-model/` -> Hugging Face Docker Space model API
- `backend/` -> Flask app for Render
- `frontend/` -> templates and static assets used by the backend UI

## Folder structure

```text
backend/
frontend/
plant-disease-model/
render.yaml
```

## Deployment flow

1. Deploy `plant-disease-model/` to Hugging Face Space.
2. Deploy the repo root to Render using `render.yaml`.
3. Point Vercel at the Render URL as the public frontend domain by using rewrites/proxying.

## Render environment variables

- `HF_MODEL_API_URL=https://vaibhav76187-plant-disease-model.hf.space/predict`
- `OPENWEATHER_API_KEY=...`
- `OPENROUTER_API_KEY=...`
- `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
- `OPENROUTER_MODEL=openai/gpt-4o-mini`

## Notes

- The `.keras` model is intentionally kept inside `plant-disease-model/`.
- If the model file grows beyond 100 MB, store it with Git LFS.
- Generated uploads and Grad-CAM images are ignored from git.

## Vercel

This repo includes [vercel.json](f:/Apps/plant/PlantCareAI/PlantCareAI/vercel.json) that rewrites all traffic to:

- `https://plantcareai-ml-app.onrender.com`

That means Vercel can be used as the public frontend URL while Render continues to run the Flask app and serve the Jinja templates safely.
